from __future__ import annotations

from enum import Enum

import pandas as pd

from bot.utils.config import RegimeSettings


class Regime(str, Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    CHAOS = "CHAOS"


class RegimeDetector:
    def __init__(self, settings: RegimeSettings) -> None:
        self.settings = settings

    def apply(self, df: pd.DataFrame) -> pd.Series:
        vol_thresh = df["realized_vol_24"].expanding().quantile(self.settings.chaos_vol_percentile)
        chaos = (df["realized_vol_24"] > vol_thresh) & (df["range_pct"] > df["range_pct"].rolling(24, min_periods=24).mean())

        trend = (
            (df[f"adx_{self.settings.adx_period}"] >= self.settings.adx_trend_threshold)
            & (df["slope_24"].abs() > 0.005)
            & (~chaos)
        )

        range_regime = (
            (df[f"adx_{self.settings.adx_period}"] < self.settings.adx_trend_threshold)
            & (df[f"bb_width_{self.settings.bb_period}"] <= self.settings.bb_width_range_threshold)
            & (~chaos)
        )

        regime = pd.Series(Regime.RANGE.value, index=df.index)
        regime.loc[chaos] = Regime.CHAOS.value
        regime.loc[trend] = Regime.TREND.value
        regime.loc[range_regime] = Regime.RANGE.value
        return regime

    def detect(self, row: pd.Series) -> Regime:
        return Regime(row["regime"])
