from __future__ import annotations

import pandas as pd

from trading_bot.backtest.engine import run_backtest
from trading_bot.config import MarketStructureConfig
from trading_bot.strategies.market_structure_hhhl import prepare_structure_features


def _df_for_engine() -> pd.DataFrame:
    vals = [100, 101, 102, 103, 104, 103, 102, 101, 100, 99, 98, 99, 100, 101, 102]
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


def test_position_sizing_and_trade_generation() -> None:
    raw = _df_for_engine()
    cfg = MarketStructureConfig(tp_mode="B", rr_target=1.5, stop_mode="swing", pivot_left=2, pivot_right=2)
    feat, _ = prepare_structure_features(raw, cfg)
    sig = pd.Series(0, index=feat.index, dtype=int)
    sig.iloc[8] = -1

    trades, equity = run_backtest(feat, sig, cfg)
    assert not equity.empty
    if not trades.empty:
        assert (trades["qty"] > 0).all()


def test_stop_and_tp_execution_paths() -> None:
    vals = [100, 101, 102, 103, 104, 103, 102, 101, 100, 101, 102, 103, 104, 105, 106]
    idx = pd.date_range("2023-01-01", periods=len(vals), freq="1h", tz="UTC")
    raw = pd.DataFrame(
        {
            "open_time": idx,
            "open": vals,
            "high": [v + 1.5 for v in vals],
            "low": [v - 1.5 for v in vals],
            "close": vals,
            "volume": 100.0,
        }
    )
    cfg = MarketStructureConfig(tp_mode="B", rr_target=1.5, stop_mode="atr", atr_mult=2.0)
    feat, _ = prepare_structure_features(raw, cfg)
    sig = pd.Series(0, index=feat.index, dtype=int)
    sig.iloc[8] = 1

    trades, _ = run_backtest(feat, sig, cfg)
    if not trades.empty:
        assert trades["exit_reason"].isin(["stop", "tp", "trailing_structure", "time_stop"]).all()
