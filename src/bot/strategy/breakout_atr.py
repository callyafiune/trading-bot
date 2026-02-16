from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot.utils.config import StrategyBreakoutSettings


Side = Literal["LONG", "SHORT"]


@dataclass
class Signal:
    side: Side
    reason: str


class BreakoutATRStrategy:
    def __init__(self, settings: StrategyBreakoutSettings) -> None:
        self.settings = settings

    def signal_at(self, df: pd.DataFrame, i: int) -> Signal | None:
        if i < self.settings.breakout_lookback_N + 1:
            return None
        row = df.iloc[i]
        if row.get("regime") != "TREND":
            return None

        if self.settings.use_rel_volume_filter and row.get("rel_volume_24", 0.0) < self.settings.min_rel_volume:
            return None

        lookback = df.iloc[i - self.settings.breakout_lookback_N : i]
        prev_high = lookback["high"].max()
        prev_low = lookback["low"].min()
        close = row["close"]

        if close > prev_high:
            return Signal(side="LONG", reason="breakout_high")
        if close < prev_low:
            return Signal(side="SHORT", reason="breakout_low")
        return None

    def initial_stop(self, side: Side, entry_price: float, atr: float) -> float:
        k = self.settings.atr_k
        return entry_price - k * atr if side == "LONG" else entry_price + k * atr

    def trailing_stop(self, side: Side, curr_stop: float, close: float, atr: float) -> float:
        k = self.settings.atr_k
        candidate = close - k * atr if side == "LONG" else close + k * atr
        if side == "LONG":
            return max(curr_stop, candidate)
        return min(curr_stop, candidate)
