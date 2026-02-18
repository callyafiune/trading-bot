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

    @staticmethod
    def _apply_hysteresis(raw: pd.Series, confirm_bars: int, initial: str) -> pd.Series:
        stable = []
        current = initial
        candidate = current
        streak = 0
        for value in raw:
            if value == current:
                candidate = current
                streak = 0
            else:
                if value == candidate:
                    streak += 1
                else:
                    candidate = value
                    streak = 1
                if streak >= confirm_bars:
                    current = candidate
                    candidate = current
                    streak = 0
            stable.append(current)
        return pd.Series(stable, index=raw.index)

    def apply(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ema200 = df["ema200"]
        slope = df["slope_ema200"]

        macro_raw = pd.Series("TRANSITION", index=df.index)
        macro_raw.loc[(close > ema200) & (slope > self.settings.macro_slope_min)] = "BULL"
        macro_raw.loc[(close < ema200) & (slope < -self.settings.macro_slope_min)] = "BEAR"
        macro = self._apply_hysteresis(macro_raw, self.settings.macro_confirm_bars, "TRANSITION")

        adx = df[f"adx_{self.settings.adx_period}"]
        chaos = (df["vol_roll"] > self.settings.chaos_vol_threshold) | (df["atr_pct"] > self.settings.chaos_atr_pct_threshold)
        micro_raw = pd.Series("RANGE", index=df.index)
        micro_raw.loc[chaos] = "CHAOS"

        trend_state = []
        in_trend = False
        for i in range(len(df)):
            if chaos.iloc[i]:
                in_trend = False
                trend_state.append("CHAOS")
                continue
            value = adx.iloc[i]
            if pd.notna(value):
                if not in_trend and value >= self.settings.adx_enter_threshold:
                    in_trend = True
                elif in_trend and value < self.settings.adx_exit_threshold:
                    in_trend = False
            trend_state.append("TREND" if in_trend else "RANGE")

        micro_raw = pd.Series(trend_state, index=df.index)
        micro_raw.loc[chaos] = "CHAOS"
        micro = self._apply_hysteresis(micro_raw, self.settings.micro_confirm_bars, "RANGE")

        final = macro + "_" + micro
        df["regime_macro"] = macro
        df["regime_micro"] = micro
        df["regime_final"] = final
        df["regime"] = final
        df["regime_switch_macro"] = (macro != macro.shift(1)).fillna(False)
        df["regime_switch_micro"] = (micro != micro.shift(1)).fillna(False)
        df["regime_switch_total"] = (final != final.shift(1)).fillna(False)
        return final

    def detect(self, row: pd.Series) -> Regime:
        value = str(row["regime"])
        if value.startswith("BULL"):
            return Regime.TREND_UP
        if value.startswith("BEAR"):
            return Regime.TREND_DOWN
        if value.endswith("CHAOS"):
            return Regime.CHAOS
        return Regime.RANGE
