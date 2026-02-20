from __future__ import annotations

import pandas as pd

from trading_bot.regime_shift_config import RegimeShiftConfig
from trading_bot.strategies.regime_shift_major_levels import detect_regime_shift_levels


def test_no_lookahead_levels_stable_when_recomputed_on_prefix() -> None:
    vals = [100, 99, 100, 101, 102, 103, 102, 101, 100, 99, 98, 99, 100]
    idx = pd.date_range("2023-01-01", periods=len(vals), freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open_time": idx,
            "open": vals,
            "high": [v + 0.5 for v in vals],
            "low": [v - 0.5 for v in vals],
            "close": vals,
            "volume": 100.0,
        }
    )
    cfg = RegimeShiftConfig(lookback_init=4, significance_mode="percent", percent_p=0.005)
    full = detect_regime_shift_levels(df, cfg)

    for i in range(7, len(df)):
        pref = detect_regime_shift_levels(df.iloc[: i + 1].copy(), cfg)
        assert str(full["regime"].iat[i]) == str(pref["regime"].iat[-1])

        full_h = full["major_high"].iat[i]
        pref_h = pref["major_high"].iat[-1]
        if pd.notna(full_h) and pd.notna(pref_h):
            assert abs(float(full_h) - float(pref_h)) < 1e-9

        full_l = full["major_low"].iat[i]
        pref_l = pref["major_low"].iat[-1]
        if pd.notna(full_l) and pd.notna(pref_l):
            assert abs(float(full_l) - float(pref_l)) < 1e-9
