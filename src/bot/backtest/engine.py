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
    notional: float


class BacktestEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.strategy = BreakoutATRStrategy(settings.strategy_breakout)
        self.risk = RiskManager(settings.risk)
        self.broker = BrokerSim(settings.frictions.fee_rate_per_side, settings.frictions.slippage_rate_per_side)
        self.last_run_diagnostics: dict[str, int | float | str | None] = {}

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        equity = self.settings.risk.account_equity_usdt
        eq_curve = []
        trades: list[dict] = []
        position: Position | None = None
        intent: Signal | None = None
        intent_idx: int | None = None

        diagnostics: dict[str, int | float | str | None] = {
            "signals_total": 0,
            "signals_blocked_regime": 0,
            "signals_blocked_risk": 0,
            "signals_blocked_killswitch": 0,
            "entries_executed": 0,
            "killswitch_events": 0,
            "first_killswitch_at": None,
        }

        day_anchor: pd.Timestamp | None = None
        week_anchor: tuple[int, int] | None = None
        equity_day_start = equity
        equity_week_start = equity
        kill_day = False
        kill_week = False

        for i in range(1, len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            ts = row["open_time"]

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
                    fill = self.broker.execute_entry(intent.side, entry_open, size.qty)
                    position = Position(intent.side, fill.qty, fill.price, stop, next_row["open_time"], fill.fee, size.notional)
                    equity -= fill.fee
                    diagnostics["entries_executed"] += 1
                else:
                    diagnostics["signals_blocked_risk"] += 1
                intent = None

            if position is not None:
                hours_in_pos = int((row["open_time"] - position.entry_time).total_seconds() // 3600)
                stop_hit = (position.side == "LONG" and row["low"] <= position.stop) or (position.side == "SHORT" and row["high"] >= position.stop)
                time_stop = hours_in_pos >= self.settings.strategy_breakout.time_stop_hours

                if stop_hit or time_stop:
                    exit_ref = position.stop if stop_hit else float(next_row["open"])
                    exit_fill = self.broker.execute_exit(position.side, exit_ref, position.qty)
                    pnl = (exit_fill.price - position.entry_price) * position.qty if position.side == "LONG" else (position.entry_price - exit_fill.price) * position.qty
                    borrow_cost = position.notional * self.settings.frictions.borrow_interest_rate_per_hour * max(hours_in_pos, 1)
                    pnl_net = pnl - position.entry_fee - exit_fill.fee - borrow_cost
                    equity += pnl_net
                    trades.append(
                        {
                            "entry_time": position.entry_time,
                            "exit_time": row["open_time"],
                            "side": position.side,
                            "entry": position.entry_price,
                            "exit": exit_fill.price,
                            "qty": position.qty,
                            "notional": position.notional,
                            "pnl_net": pnl_net,
                            "hours_in_pos": hours_in_pos,
                            "exit_reason": "stop" if stop_hit else "time_stop",
                        }
                    )
                    position = None
                else:
                    position.stop = self.strategy.trailing_stop(position.side, position.stop, float(row["close"]), float(row["atr_14"]))

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
                sig, blocked_reason = self.strategy.signal_decision(df, i)
                if blocked_reason == "regime":
                    diagnostics["signals_blocked_regime"] += 1

                if sig:
                    diagnostics["signals_total"] += 1
                    if kill_day or kill_week:
                        diagnostics["signals_blocked_killswitch"] += 1
                    else:
                        intent = sig
                        intent_idx = i

            eq_curve.append(equity)

        self.last_run_diagnostics = diagnostics
        return pd.DataFrame(trades), pd.Series(eq_curve, name="equity")
