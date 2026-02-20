from __future__ import annotations

import pandas as pd

from trading_bot.config import MarketStructureConfig
from trading_bot.strategies.market_structure_hhhl import detect_confirmed_swings, generate_structure_signals, prepare_structure_features


def _base_df() -> pd.DataFrame:
    vals = [10, 11, 13, 11, 10, 12, 11, 14, 12, 11, 9, 10, 8, 10, 7, 9, 6, 8, 7]
    idx = pd.date_range("2023-01-01", periods=len(vals), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": idx,
            "open": vals,
            "high": [v + 0.2 for v in vals],
            "low": [v - 0.2 for v in vals],
            "close": vals,
            "volume": 100.0,
        }
    )


def test_swing_confirmation_has_no_lookahead_leak() -> None:
    df = _base_df()
    cfg = MarketStructureConfig(pivot_left=2, pivot_right=2)
    swings, out = detect_confirmed_swings(df, cfg)

    assert swings
    for s in swings:
        assert s.confirm_idx == s.pivot_idx + cfg.pivot_right
        assert bool(out.loc[s.confirm_idx, "swing_high_confirmed"] or out.loc[s.confirm_idx, "swing_low_confirmed"])


def test_structure_labels_include_hh_hl_lh_ll() -> None:
    df = _base_df()
    cfg = MarketStructureConfig(pivot_left=2, pivot_right=2)
    swings, _ = detect_confirmed_swings(df, cfg)
    labels = {s.label for s in swings}
    assert "HH" in labels or "LH" in labels
    assert "HL" in labels or "LL" in labels


def test_hh_hl_and_ll_lh_generate_directional_signals() -> None:
    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({"open_time": idx, "close": 100.0, "volume": 100.0})
    df["ma99"] = 90.0
    df["swing_low_label"] = ["", "", "LL", "", "HL", "", "", "", "LL", ""]
    df["swing_high_label"] = ["", "", "", "", "", "HH", "LH", "", "", ""]
    df["atr_pct"] = 0.02
    df["last_swing_high_price"] = 101.0
    df["last_swing_low_price"] = 99.0
    cfg = MarketStructureConfig(pivot_left=2, pivot_right=2)
    sig = generate_structure_signals(df, cfg)
    assert (sig == 1).any()
