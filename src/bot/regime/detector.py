from __future__ import annotations

from enum import Enum

import pandas as pd

from bot.utils.config import RegimeSettings


class Regime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    CHAOS = "CHAOS"


class RegimeDetector:
    def __init__(self, settings: RegimeSettings) -> None:
        self.settings = settings

    def apply(self, df: pd.DataFrame) -> pd.Series:
        vol_thresh = df["realized_vol_24"].expanding().quantile(self.settings.chaos_vol_percentile)
        chaos = (df["realized_vol_24"] > vol_thresh) & (df["atr_pct"] > self.settings.chaos_atr_pct_threshold)
        adx = df[f"adx_{self.settings.adx_period}"]
        ma200 = df["ma_200"]
        close = df["close"]
        regime = pd.Series(Regime.RANGE.value, index=df.index)
        regime.loc[chaos] = Regime.CHAOS.value

        trend_mask = (~chaos) & (adx >= self.settings.adx_trend_threshold)
        regime.loc[trend_mask & (close > ma200)] = Regime.TREND_UP.value
        regime.loc[trend_mask & (close < ma200)] = Regime.TREND_DOWN.value

        return regime

    def detect(self, row: pd.Series) -> Regime:
        return Regime(row["regime"])
