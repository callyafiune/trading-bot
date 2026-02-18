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
    entry_type: str
    trailing_active: bool = False


@dataclass
class ExitPlan:
    exit_ref: float
    exit_time: pd.Timestamp
    stop_hit: bool
    soft_time_stop: bool
    hard_time_stop: bool


class BacktestEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.strategy = BreakoutATRStrategy(
            settings.strategy_breakout,
            settings.strategy_router,
            settings.router,
            settings.funding_filter,
            settings.fng_filter,
            settings.multi_timeframe,
        )
        self.risk = RiskManager(settings.risk)
        self.broker = BrokerSim(settings.frictions.fee_rate_per_side, settings.frictions.slippage_rate_per_side)
        self.last_run_diagnostics: dict[str, int | float | str | None] = {}

    @staticmethod
    def _r_multiple(side: str, entry_price: float, price: float, denom: float) -> float:
        if side == "LONG":
            return (price - entry_price) / denom
        return (entry_price - price) / denom

    @staticmethod
    def _pnl(side: str, entry_price: float, exit_price: float, qty: float) -> float:
        if side == "LONG":
            return (exit_price - entry_price) * qty
        return (entry_price - exit_price) * qty

    def _risk_multiplier(self, row: pd.Series, side: str) -> float:
        _ = side
        if not self.settings.router.enabled:
            return 1.0
        regime = str(row.get("regime", ""))
        micro = regime.split("_")[-1] if "_" in regime else regime
        if micro == "RANGE":
            return max(0.0, float(self.settings.router.range_risk_multiplier))
        return 1.0

    def _update_stop(self, position: Position, close_price: float, atr: float, diagnostics: dict[str, int | float | str | None]) -> None:
        denom = position.initial_r if position.initial_r > 0 else 1e-12
        current_r = self._r_multiple(position.side, position.entry_price, close_price, denom)

        if self.settings.adaptive_trailing.enabled:
            if current_r >= self.settings.adaptive_trailing.activate_after_R:
                if not position.trailing_active:
                    diagnostics["adaptive_trailing_triggered"] += 1
                position.trailing_active = True
                atr_mult = self.settings.adaptive_trailing.trailing_atr_multiplier
                candidate = close_price - atr * atr_mult if position.side == "LONG" else close_price + atr * atr_mult
                if position.side == "LONG":
                    position.stop = max(position.stop, candidate)
                else:
                    position.stop = min(position.stop, candidate)
            return

        position.stop = self.strategy.trailing_stop(position.side, position.stop, close_price, atr)

    def _time_stop_flags(self, hours_in_pos: int, current_r: float) -> tuple[bool, bool]:
        if self.settings.time_exit.enabled:
            hard_time_stop = hours_in_pos >= self.settings.time_exit.max_holding_hours
            soft_time_stop = (
                hours_in_pos >= self.settings.time_exit.soft_exit_hours
                and current_r < self.settings.time_exit.min_r_multiple_after_soft
            )
            return hard_time_stop, soft_time_stop
        return hours_in_pos >= self.settings.strategy_breakout.time_stop_hours, False

    def _pick_detail_candidate(
        self,
        candidates: list[ExitPlan],
        side: str,
        entry_price: float,
        qty: float,
    ) -> ExitPlan:
        if len(candidates) == 1:
            return candidates[0]

        policy = self.settings.execution.detail_timeframe.policy
        scored = []
        for c in candidates:
            pnl = self._pnl(side, entry_price, c.exit_ref, qty)
            scored.append((pnl, c))
        if policy == "optimistic":
            return max(scored, key=lambda x: x[0])[1]
        return min(scored, key=lambda x: x[0])[1]

    def _build_trade(
        self,
        position: Position,
        exit_plan: ExitPlan,
        exit_fill_price: float,
        exit_fee: float,
        exit_slippage: float,
        trade_id_seq: int,
        diagnostics: dict[str, int | float | str | None],
    ) -> tuple[dict, float]:
        hours_in_pos = int(max(0, (exit_plan.exit_time - position.entry_time).total_seconds() // 3600))
        denom = position.initial_r if position.initial_r > 0 else 1e-12
        pnl = self._pnl(position.side, position.entry_price, exit_fill_price, position.qty)
        r_multiple_exit = self._r_multiple(position.side, position.entry_price, exit_fill_price, denom)
        borrow_cost = position.notional * self.settings.frictions.borrow_interest_rate_per_hour * max(hours_in_pos, 1)
        fees_total = position.entry_fee + exit_fee
        slippage_total = position.entry_slippage + exit_slippage
        pnl_net = pnl - fees_total - borrow_cost

        if exit_plan.stop_hit:
            exit_reason = "trailing_stop" if position.trailing_active else "atr_stop"
            if position.trailing_active:
                diagnostics["adaptive_trailing_stop_hits"] += 1
            reason_exit = "STOP"
        elif exit_plan.soft_time_stop:
            exit_reason = "soft_time_stop"
            diagnostics["time_exit_triggered"] += 1
            reason_exit = "TIME"
        else:
            exit_reason = "time_stop"
            if exit_plan.hard_time_stop:
                diagnostics["time_exit_triggered"] += 1
            reason_exit = "TIME"

        trade = {
            "trade_id": trade_id_seq,
            "entry_time": position.entry_time,
            "exit_time": exit_plan.exit_time,
            "direction": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_fill_price,
            "qty": position.qty,
            "notional": position.notional,
            "stop_init": position.stop_init,
            "stop_final": position.stop,
            "reason_exit": reason_exit,
            "exit_reason": exit_reason,
            "entry_type": position.entry_type,
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
        return trade, pnl_net

    def _detail_exit_plan(
        self,
        position: Position,
        row: pd.Series,
        next_row: pd.Series,
        detail_slice: pd.DataFrame,
        diagnostics: dict[str, int | float | str | None],
    ) -> ExitPlan | None:
        atr = float(row.get("atr_14", 0.0))
        for _, drow in detail_slice.iterrows():
            ts = drow["open_time"]
            high = float(drow["high"])
            low = float(drow["low"])
            close = float(drow["close"])
            denom = position.initial_r if position.initial_r > 0 else 1e-12
            current_r = self._r_multiple(position.side, position.entry_price, close, denom)
            hours_in_pos = int(max(0, (ts - position.entry_time).total_seconds() // 3600))
            hard_time_stop, soft_time_stop = self._time_stop_flags(hours_in_pos, current_r)
            stop_hit = (position.side == "LONG" and low <= position.stop) or (position.side == "SHORT" and high >= position.stop)

            candidates: list[ExitPlan] = []
            if stop_hit:
                candidates.append(ExitPlan(position.stop, ts, True, False, False))
            if hard_time_stop:
                candidates.append(ExitPlan(close, ts, False, False, True))
            if soft_time_stop:
                candidates.append(ExitPlan(close, ts, False, True, False))

            if candidates:
                return self._pick_detail_candidate(candidates, position.side, position.entry_price, position.qty)

            self._update_stop(position, close, atr, diagnostics)

        if detail_slice.empty:
            return None
        # If no detail exit happened, keep using HTF close for eventual trailing progression.
        self._update_stop(position, float(row["close"]), atr, diagnostics)
        _ = next_row
        return None

    def run(self, df: pd.DataFrame, detail_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.strategy.prepare(df)
        detail_enabled = bool(self.settings.execution.detail_timeframe.enabled and detail_df is not None and not detail_df.empty)
        if detail_enabled:
            detail_df = detail_df.sort_values("open_time").reset_index(drop=True)
        else:
            detail_df = None

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
            "blocked_funding_long": 0,
            "blocked_funding_short": 0,
            "blocked_funding_total": 0,
            "blocked_fng_long": 0,
            "blocked_fng_short": 0,
            "blocked_fng_total": 0,
            "blocked_macro": 0,
            "blocked_micro": 0,
            "blocked_chaos": 0,
            "blocked_cooldown": 0,
            "blocked_mtf": 0,
            "time_exit_triggered": 0,
            "adaptive_trailing_triggered": 0,
            "adaptive_trailing_stop_hits": 0,
            "detail_timeframe_enabled": int(detail_enabled),
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
                risk_mult = self._risk_multiplier(row, intent.side)
                size = self.risk.size_position(
                    equity,
                    entry_open,
                    stop,
                    intent.side,
                    risk_multiplier=risk_mult,
                )
                if size.valid and size.qty > 0:
                    qty = size.qty
                    if qty <= 0:
                        diagnostics["signals_blocked_risk"] += 1
                        intent = None
                        continue
                    fill = self.broker.execute_entry(intent.side, entry_open, qty)
                    notional = fill.price * fill.qty
                    position = Position(
                        side=intent.side,
                        qty=fill.qty,
                        entry_price=fill.price,
                        stop=stop,
                        entry_time=next_row["open_time"],
                        entry_fee=fill.fee,
                        entry_slippage=fill.slippage,
                        notional=notional,
                        stop_init=stop,
                        regime_at_entry=str(row.get("regime", "")),
                        initial_r=abs(fill.price - stop),
                        entry_type=intent.entry_type,
                        trailing_active=False,
                    )
                    equity -= fill.fee
                    diagnostics["entries_executed"] += 1
                    entered_now = True
                else:
                    diagnostics["signals_blocked_risk"] += 1
                intent = None

            if position is not None and not entered_now:
                exit_plan: ExitPlan | None = None
                if detail_enabled and detail_df is not None:
                    detail_slice = detail_df[
                        (detail_df["open_time"] >= row["open_time"])
                        & (detail_df["open_time"] < next_row["open_time"])
                    ]
                    exit_plan = self._detail_exit_plan(position, row, next_row, detail_slice, diagnostics)

                if exit_plan is None:
                    close_price = float(row["close"])
                    denom = position.initial_r if position.initial_r > 0 else 1e-12
                    current_r = self._r_multiple(position.side, position.entry_price, close_price, denom)
                    hours_in_pos = int(max(0, (row["open_time"] - position.entry_time).total_seconds() // 3600))
                    hard_time_stop, soft_time_stop = self._time_stop_flags(hours_in_pos, current_r)
                    stop_hit = (
                        (position.side == "LONG" and row["low"] <= position.stop)
                        or (position.side == "SHORT" and row["high"] >= position.stop)
                    )
                    if stop_hit:
                        exit_plan = ExitPlan(position.stop, row["open_time"], True, False, False)
                    elif hard_time_stop:
                        exit_plan = ExitPlan(float(next_row["open"]), row["open_time"], False, False, True)
                    elif soft_time_stop:
                        exit_plan = ExitPlan(float(next_row["open"]), row["open_time"], False, True, False)
                    else:
                        self._update_stop(position, close_price, float(row["atr_14"]), diagnostics)

                if exit_plan is not None:
                    exit_fill = self.broker.execute_exit(position.side, float(exit_plan.exit_ref), position.qty)
                    trade, pnl_net = self._build_trade(
                        position=position,
                        exit_plan=exit_plan,
                        exit_fill_price=exit_fill.price,
                        exit_fee=exit_fill.fee,
                        exit_slippage=exit_fill.slippage,
                        trade_id_seq=trade_id_seq,
                        diagnostics=diagnostics,
                    )
                    equity += pnl_net
                    trades.append(trade)
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
                elif blocked_reason == "funding_long":
                    diagnostics["blocked_funding"] += 1
                    diagnostics["blocked_funding_long"] += 1
                    diagnostics["blocked_funding_total"] += 1
                elif blocked_reason == "funding_short":
                    diagnostics["blocked_funding"] += 1
                    diagnostics["blocked_funding_short"] += 1
                    diagnostics["blocked_funding_total"] += 1
                elif blocked_reason == "fng_long":
                    diagnostics["blocked_fng_long"] += 1
                    diagnostics["blocked_fng_total"] += 1
                elif blocked_reason == "fng_short":
                    diagnostics["blocked_fng_short"] += 1
                    diagnostics["blocked_fng_total"] += 1
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
