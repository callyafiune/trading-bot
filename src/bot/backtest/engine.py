from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bot.execution.broker_sim import BrokerSim
from bot.risk.manager import RiskManager
from bot.strategy.breakout_atr import BreakoutATRStrategy, Signal
from bot.utils.config import Settings


@dataclass
class Position:
    side: str
    qty: float
    entry_price: float
    stop: float
    entry_time: pd.Timestamp
    entry_fee: float
    entry_slippage: float
    notional: float
    stop_init: float
    regime_at_entry: str
    initial_r: float
    trailing_active: bool = False


class BacktestEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.strategy = BreakoutATRStrategy(
            settings.strategy_breakout,
            settings.strategy_router,
            settings.funding_filter,
            settings.multi_timeframe,
        )
        self.risk = RiskManager(settings.risk)
        self.broker = BrokerSim(settings.frictions.fee_rate_per_side, settings.frictions.slippage_rate_per_side)
        self.last_run_diagnostics: dict[str, int | float | str | None] = {}

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.strategy.prepare(df)
        equity = self.settings.risk.account_equity_usdt
        eq_curve = []
        trades: list[dict] = []
        position: Position | None = None
        intent: Signal | None = None
        intent_idx: int | None = None
        cooldown_until_idx: dict[str, int] = {"LONG": -1, "SHORT": -1}

        diagnostics: dict[str, int | float | str | None] = {
            "signals_total": 0,
            "signals_blocked_regime": 0,
            "signals_blocked_risk": 0,
            "signals_blocked_killswitch": 0,
            "entries_executed": 0,
            "killswitch_events": 0,
            "first_killswitch_at": None,
            "signals_blocked_mode": 0,
            "signals_blocked_ma200": 0,
            "blocked_by_regime_reason": {},
            "blocked_funding": 0,
            "blocked_macro": 0,
            "blocked_micro": 0,
            "blocked_chaos": 0,
            "blocked_cooldown": 0,
            "blocked_mtf": 0,
            "time_exit_triggered": 0,
            "adaptive_trailing_triggered": 0,
            "adaptive_trailing_stop_hits": 0,
        }

        day_anchor: pd.Timestamp | None = None
        week_anchor: tuple[int, int] | None = None
        equity_day_start = equity
        equity_week_start = equity
        kill_day = False
        kill_week = False
        trade_id_seq = 1

        for i in range(1, len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            ts = row["open_time"]
            entered_now = False

            iso = ts.isocalendar()
            curr_day = ts.date()
            curr_week = (int(iso.year), int(iso.week))
            if day_anchor is None or curr_day != day_anchor.date():
                day_anchor = ts
                equity_day_start = equity
                kill_day = False
            if week_anchor is None or curr_week != week_anchor:
                week_anchor = curr_week
                equity_week_start = equity
                kill_week = False

            if intent is not None and position is None and intent_idx == i - 1:
                entry_open = float(next_row["open"])
                atr = float(row["atr_14"])
                stop = self.strategy.initial_stop(intent.side, entry_open, atr)
                size = self.risk.size_position(equity, entry_open, stop, intent.side)
                if size.valid and size.qty > 0:
                    qty = size.qty
                    funding_action = str(row.get("funding_action", "none"))
                    if funding_action == "reduce_size":
                        qty = qty * self.settings.funding_filter.reduce_size_factor
                    if qty <= 0:
                        diagnostics["signals_blocked_risk"] += 1
                        intent = None
                        continue
                    fill = self.broker.execute_entry(intent.side, entry_open, qty)
                    position = Position(
                        intent.side,
                        fill.qty,
                        fill.price,
                        stop,
                        next_row["open_time"],
                        fill.fee,
                        fill.slippage,
                        size.notional,
                        stop,
                        str(row.get("regime", "")),
                        abs(fill.price - stop),
                        False,
                    )
                    equity -= fill.fee
                    diagnostics["entries_executed"] += 1
                    entered_now = True
                else:
                    diagnostics["signals_blocked_risk"] += 1
                intent = None

            if position is not None and not entered_now:
                hours_in_pos = int((row["open_time"] - position.entry_time).total_seconds() // 3600)
                stop_hit = (position.side == "LONG" and row["low"] <= position.stop) or (position.side == "SHORT" and row["high"] >= position.stop)
                close_price = float(row["close"])
                denom = position.initial_r if position.initial_r > 0 else 1e-12
                current_r = (
                    (close_price - position.entry_price) / denom
                    if position.side == "LONG"
                    else (position.entry_price - close_price) / denom
                )

                if self.settings.adaptive_trailing.enabled:
                    if current_r >= self.settings.adaptive_trailing.activate_after_R:
                        if not position.trailing_active:
                            diagnostics["adaptive_trailing_triggered"] += 1
                        position.trailing_active = True
                        atr_mult = self.settings.adaptive_trailing.trailing_atr_multiplier
                        candidate = (
                            close_price - float(row["atr_14"]) * atr_mult
                            if position.side == "LONG"
                            else close_price + float(row["atr_14"]) * atr_mult
                        )
                        if position.side == "LONG":
                            position.stop = max(position.stop, candidate)
                        else:
                            position.stop = min(position.stop, candidate)
                else:
                    position.stop = self.strategy.trailing_stop(position.side, position.stop, close_price, float(row["atr_14"]))

                if self.settings.time_exit.enabled:
                    hard_time_stop = hours_in_pos >= self.settings.time_exit.max_holding_hours
                    soft_time_stop = (
                        hours_in_pos >= self.settings.time_exit.soft_exit_hours
                        and current_r < self.settings.time_exit.min_r_multiple_after_soft
                    )
                else:
                    hard_time_stop = hours_in_pos >= self.settings.strategy_breakout.time_stop_hours
                    soft_time_stop = False

                if stop_hit or hard_time_stop or soft_time_stop:
                    exit_ref = position.stop if stop_hit else float(next_row["open"])
                    exit_fill = self.broker.execute_exit(position.side, exit_ref, position.qty)
                    pnl = (exit_fill.price - position.entry_price) * position.qty if position.side == "LONG" else (position.entry_price - exit_fill.price) * position.qty
                    r_multiple_exit = (
                        (exit_fill.price - position.entry_price) / denom
                        if position.side == "LONG"
                        else (position.entry_price - exit_fill.price) / denom
                    )
                    borrow_cost = position.notional * self.settings.frictions.borrow_interest_rate_per_hour * max(hours_in_pos, 1)
                    fees_total = position.entry_fee + exit_fill.fee
                    slippage_total = position.entry_slippage + exit_fill.slippage
                    pnl_net = pnl - fees_total - borrow_cost
                    equity += pnl_net
                    if stop_hit:
                        exit_reason = "trailing_stop" if position.trailing_active else "atr_stop"
                        if position.trailing_active:
                            diagnostics["adaptive_trailing_stop_hits"] += 1
                    elif soft_time_stop:
                        exit_reason = "soft_time_stop"
                    else:
                        exit_reason = "time_stop"
                        diagnostics["time_exit_triggered"] += 1
                    if soft_time_stop:
                        diagnostics["time_exit_triggered"] += 1
                    trades.append(
                        {
                            "trade_id": trade_id_seq,
                            "entry_time": position.entry_time,
                            "exit_time": row["open_time"],
                            "direction": position.side,
                            "entry_price": position.entry_price,
                            "exit_price": exit_fill.price,
                            "qty": position.qty,
                            "notional": position.notional,
                            "stop_init": position.stop_init,
                            "stop_final": position.stop,
                            "reason_exit": "STOP" if stop_hit else "TIME",
                            "exit_reason": exit_reason,
                            "pnl_gross": pnl,
                            "pnl_net": pnl_net,
                            "fees": fees_total,
                            "slippage": slippage_total,
                            "interest": borrow_cost,
                            "regime_at_entry": position.regime_at_entry,
                            "hold_hours": hours_in_pos,
                            "holding_hours": hours_in_pos,
                            "R_multiple_exit": r_multiple_exit,
                        }
                    )
                    trade_id_seq += 1
                    cooldown_bars = max(0, int(self.settings.strategy_router.cooldown_bars_after_exit))
                    cooldown_until_idx[position.side] = i + cooldown_bars
                    position = None

            daily_pnl_pct = (equity - equity_day_start) / equity_day_start if equity_day_start else 0.0
            weekly_pnl_pct = (equity - equity_week_start) / equity_week_start if equity_week_start else 0.0

            if not kill_day and daily_pnl_pct <= -self.settings.risk.daily_loss_limit_pct:
                kill_day = True
                diagnostics["killswitch_events"] += 1
                if diagnostics["first_killswitch_at"] is None:
                    diagnostics["first_killswitch_at"] = str(ts)
            if not kill_week and weekly_pnl_pct <= -self.settings.risk.weekly_loss_limit_pct:
                kill_week = True
                diagnostics["killswitch_events"] += 1
                if diagnostics["first_killswitch_at"] is None:
                    diagnostics["first_killswitch_at"] = str(ts)

            if position is None:
                decision = self.strategy.evaluate_signal(df, i)
                sig = decision.signal
                blocked_reason = decision.blocked_reason
                if decision.raw_signal is not None:
                    diagnostics["signals_total"] += 1

                if blocked_reason and blocked_reason.startswith("blocked_"):
                    diagnostics["signals_blocked_regime"] += 1
                    blocked_by_regime = diagnostics["blocked_by_regime_reason"]
                    assert isinstance(blocked_by_regime, dict)
                    blocked_by_regime[blocked_reason] = int(blocked_by_regime.get(blocked_reason, 0)) + 1
                    if "trend_" in blocked_reason:
                        diagnostics["blocked_macro"] += 1
                    if "range" in blocked_reason:
                        diagnostics["blocked_micro"] += 1
                    if "chaos" in blocked_reason:
                        diagnostics["blocked_chaos"] += 1
                elif blocked_reason == "funding":
                    diagnostics["blocked_funding"] += 1
                elif blocked_reason == "ma200":
                    diagnostics["signals_blocked_ma200"] += 1
                elif blocked_reason == "mtf":
                    diagnostics["blocked_mtf"] += 1
                elif blocked_reason in ("macd_gate", "ml_gate", "unsupported_mode"):
                    diagnostics["signals_blocked_mode"] += 1

                if sig:
                    if kill_day or kill_week:
                        diagnostics["signals_blocked_killswitch"] += 1
                    elif i <= cooldown_until_idx.get(sig.side, -1):
                        diagnostics["blocked_cooldown"] += 1
                    else:
                        intent = sig
                        intent_idx = i

            current_position = "FLAT" if position is None else position.side
            close_price = float(row["close"])
            eq_curve.append(
                {
                    "timestamp": row["open_time"],
                    "equity": equity,
                    "position": current_position,
                    "price": close_price,
                }
            )

        self.last_run_diagnostics = diagnostics
        equity_df = pd.DataFrame(eq_curve)
        if not equity_df.empty:
            running_peak = equity_df["equity"].cummax()
            equity_df["drawdown"] = equity_df["equity"] / running_peak - 1.0
        else:
            equity_df["drawdown"] = pd.Series(dtype=float)
        return pd.DataFrame(trades), equity_df
