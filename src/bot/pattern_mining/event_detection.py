from __future__ import annotations

import pandas as pd


def detect_events(
    df: pd.DataFrame,
    big_lower_wick_threshold: float = 0.6,
    sweep_lookback: int = 20,
    structure_lookback: int = 20,
) -> pd.DataFrame:
    out = df.copy()

    prev_sweep_low = out["low"].shift(1).rolling(sweep_lookback, min_periods=sweep_lookback).min()
    prev_struct_high = out["high"].shift(1).rolling(structure_lookback, min_periods=structure_lookback).max()
    prev_struct_low = out["low"].shift(1).rolling(structure_lookback, min_periods=structure_lookback).min()

    out["big_lower_wick"] = out["lower_wick_ratio"] >= big_lower_wick_threshold
    out["sweep_down_reclaim"] = (out["low"] < prev_sweep_low) & (out["close"] > prev_sweep_low)
    out["break_structure_up"] = out["close"] > prev_struct_high
    out["break_structure_down"] = out["close"] < prev_struct_low
    out["regime_above_ma99"] = out["close"] > out["ma99"]
    out["regime_ma99_slope_up"] = out["slope_ma99"] > 0

    return out
