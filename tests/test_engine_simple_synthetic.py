from __future__ import annotations

import pandas as pd

from trading_bot.backtest.regime_shift_engine import run_regime_shift_backtest
from trading_bot.regime_shift_config import RegimeShiftConfig
from trading_bot.strategies.regime_shift_major_levels import detect_regime_shift_levels


def _mk_df(vals: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=len(vals), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": idx,
            "open": vals,
            "high": [v + 0.6 for v in vals],
            "low": [v - 0.6 for v in vals],
            "close": vals,
            "volume": 100.0,
        }
    )


def test_flip_closes_and_reverses_position() -> None:
    vals = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 96, 97]
    df = _mk_df(vals)
    cfg = RegimeShiftConfig(
        lookback_init=4,
        significance_mode="percent",
        percent_p=0.003,
        break_confirm_mode="close_only",
        allow_flip=True,
        tp_mode="trailing_major",
    )
    feat = detect_regime_shift_levels(df, cfg)
    trades, _ = run_regime_shift_backtest(feat, cfg)
    assert not trades.empty
    assert (trades["side"] == "LONG").any()
    assert (trades["side"] == "SHORT").any()


def test_significance_modes_both_generate_trades_on_short_dataset() -> None:
    vals = [100, 99, 98, 97, 98, 99, 100, 101, 102, 101, 100, 99, 98, 97, 98, 99, 100, 101]
    df = _mk_df(vals)

    cfg_atr = RegimeShiftConfig(lookback_init=4, significance_mode="atr", atr_k=1.5, break_confirm_mode="close_only")
    cfg_pct = RegimeShiftConfig(lookback_init=4, significance_mode="percent", percent_p=0.005, break_confirm_mode="close_only")

    tr_atr, _ = run_regime_shift_backtest(detect_regime_shift_levels(df, cfg_atr), cfg_atr)
    tr_pct, _ = run_regime_shift_backtest(detect_regime_shift_levels(df, cfg_pct), cfg_pct)

    assert len(tr_atr) > 0
    assert len(tr_pct) > 0
