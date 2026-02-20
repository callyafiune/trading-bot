from __future__ import annotations

import pandas as pd

from trading_bot.regime_shift_config import RegimeShiftConfig
from trading_bot.strategies.regime_shift_major_levels import detect_regime_shift_levels


def test_break_confirm_n_closes_enters_on_expected_candle() -> None:
    vals = [100, 100, 100, 100, 101, 101, 102, 103, 104]
    idx = pd.date_range("2023-01-01", periods=len(vals), freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open_time": idx,
            "open": vals,
            "high": [v + 0.4 for v in vals],
            "low": [v - 0.4 for v in vals],
            "close": vals,
            "volume": 100.0,
        }
    )
    cfg = RegimeShiftConfig(
        lookback_init=3,
        significance_mode="percent",
        percent_p=0.004,
        break_confirm_mode="n_closes",
        n_closes_confirm=2,
        buffer_mode="percent",
        percent_buffer_p=0.0,
    )
    out = detect_regime_shift_levels(df, cfg)
    shifts = out.index[out["bullish_shift"]].tolist()
    assert shifts, "Expected at least one bullish_shift"
    first_shift = shifts[0]
    assert first_shift == 5
    assert out["close"].iat[first_shift] > out["major_high"].shift(1).iat[first_shift]
