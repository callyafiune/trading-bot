from __future__ import annotations

import pandas as pd

from trading_bot.regime_shift_config import RegimeShiftConfig
from trading_bot.strategies.regime_shift_major_levels import detect_regime_shift_levels


def _mk_df(vals: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=len(vals), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": idx,
            "open": vals,
            "high": [v + 0.5 for v in vals],
            "low": [v - 0.5 for v in vals],
            "close": vals,
            "volume": 100.0,
        }
    )


def test_major_levels_trailing_in_bullish_trend() -> None:
    vals = [100, 101, 102, 103, 104, 106, 107, 109, 108, 110, 111, 112, 111, 113, 114]
    df = _mk_df(vals)
    cfg = RegimeShiftConfig(lookback_init=4, significance_mode="percent", percent_p=0.003, break_confirm_mode="close_only")
    out = detect_regime_shift_levels(df, cfg)

    bull = out[out["regime"] == "bullish"]
    assert len(bull) >= 3
    diffs = bull["major_low"].ffill().diff().dropna()
    assert (diffs >= -1e-9).all()

    mh_diffs = bull["major_high"].ffill().diff().dropna()
    assert (mh_diffs >= -1e-9).all()


def test_break_confirmation_with_buffer() -> None:
    vals = [100, 100, 100, 100, 101, 102, 103, 104, 105, 106]
    df = _mk_df(vals)
    cfg = RegimeShiftConfig(
        lookback_init=3,
        significance_mode="percent",
        percent_p=0.003,
        break_confirm_mode="close_plus_buffer",
        buffer_mode="percent",
        percent_buffer_p=0.001,
    )
    out = detect_regime_shift_levels(df, cfg)
    assert out["bullish_shift"].any()


def test_no_lookahead_major_levels_incremental() -> None:
    vals = [100, 99, 98, 97, 98, 99, 100, 101, 100, 99, 98, 97, 96]
    df = _mk_df(vals)
    cfg = RegimeShiftConfig(lookback_init=4, significance_mode="percent", percent_p=0.005)
    full = detect_regime_shift_levels(df, cfg)

    for i in range(6, len(df)):
        sub = detect_regime_shift_levels(df.iloc[: i + 1].copy(), cfg)
        assert str(full["regime"].iat[i]) == str(sub["regime"].iat[-1])
        a = full["major_high"].iat[i]
        b = sub["major_high"].iat[-1]
        if pd.notna(a) and pd.notna(b):
            assert abs(float(a) - float(b)) < 1e-9
