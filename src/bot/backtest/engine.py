from __future__ import annotations

from dataclasses import dataclass, asdict

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

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        equity = self.settings.risk.account_equity_usdt
        eq_curve = []
        trades: list[dict] = []
        position: Position | None = None
        intent: Signal | None = None
        intent_idx: int | None = None

        for i in range(1, len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]

            if intent is not None and position is None and intent_idx == i - 1:
                entry_open = float(next_row["open"])
                atr = float(row["atr_14"])
                stop = self.strategy.initial_stop(intent.side, entry_open, atr)
                size = self.risk.size_position(equity, entry_open, stop, intent.side)
                if size.valid and size.qty > 0:
                    fill = self.broker.execute_entry(intent.side, entry_open, size.qty)
                    position = Position(intent.side, fill.qty, fill.price, stop, next_row["open_time"], fill.fee, size.notional)
                    equity -= fill.fee
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

            daily_pnl_pct = (equity - self.settings.risk.account_equity_usdt) / self.settings.risk.account_equity_usdt
            weekly_pnl_pct = daily_pnl_pct
            if position is None and not self.risk.hit_kill_switch(daily_pnl_pct, weekly_pnl_pct):
                sig = self.strategy.signal_at(df, i)
                if sig:
                    intent = sig
                    intent_idx = i

            eq_curve.append(equity)

        return pd.DataFrame(trades), pd.Series(eq_curve, name="equity")
