import numpy as np
import pandas as pd

from bot.features.builder import build_features
from bot.market_data.loader import merge_ohlcv_with_funding, process_funding_to_1h
from bot.regime.detector import RegimeDetector
from bot.utils.config import FeatureSettings, RegimeSettings


def _base_df(n: int = 320) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.Series(np.linspace(100, 140, n), index=idx)
    return pd.DataFrame(
        {
            "open_time": idx,
            "open": close.values,
            "high": close.values + 1,
            "low": close.values - 1,
            "close": close.values,
            "volume": 1000.0,
        }
    )


def test_no_lookahead_ema_and_slope() -> None:
    df = _base_df()
    features = build_features(df, FeatureSettings(ema_slow=200, slope_window=10))

    mutated = df.copy()
    mutated.loc[mutated.index[-1], "close"] = mutated.loc[mutated.index[-1], "close"] * 10
    changed = build_features(mutated, FeatureSettings(ema_slow=200, slope_window=10))

    assert features["ema200"].iloc[:-1].equals(changed["ema200"].iloc[:-1])
    assert features["slope_ema200"].iloc[:-1].equals(changed["slope_ema200"].iloc[:-1])


def test_macro_regime_classification() -> None:
    df = _base_df()
    feat = build_features(df, FeatureSettings())
    feat["adx_14"] = 35.0
    detector = RegimeDetector(RegimeSettings(macro_confirm_bars=1, micro_confirm_bars=1, chaos_vol_threshold=1.0))

    bull = feat.copy()
    bull["close"] = bull["ema200"] * 1.01
    bull["slope_ema200"] = 0.01
    out_bull = detector.apply(bull)
    assert out_bull.iloc[-1].startswith("BULL")

    bear = feat.copy()
    bear["close"] = bear["ema200"] * 0.99
    bear["slope_ema200"] = -0.01
    out_bear = detector.apply(bear)
    assert out_bear.iloc[-1].startswith("BEAR")


def test_hysteresis_reduces_switches() -> None:
    n = 200
    idx = pd.RangeIndex(n)
    close = pd.Series(np.where(np.arange(n) % 2 == 0, 101.0, 99.0), index=idx)
    df = pd.DataFrame(
        {
            "close": close,
            "ema200": 100.0,
            "slope_ema200": np.where(np.arange(n) % 2 == 0, 0.01, -0.01),
            "vol_roll": 0.01,
            "atr_pct": 0.01,
            "adx_14": 26.0,
        }
    )
    no_hyst = RegimeDetector(
        RegimeSettings(macro_confirm_bars=1, micro_confirm_bars=1, adx_enter_threshold=28, adx_exit_threshold=24)
    )
    with_hyst = RegimeDetector(
        RegimeSettings(macro_confirm_bars=6, micro_confirm_bars=3, adx_enter_threshold=28, adx_exit_threshold=24)
    )
    out_no = no_hyst.apply(df.copy())
    out_yes = with_hyst.apply(df.copy())
    switches_no = int((out_no != out_no.shift(1)).sum())
    switches_yes = int((out_yes != out_yes.shift(1)).sum())
    assert switches_yes < switches_no


def test_funding_merge_alignment() -> None:
    ohlcv = _base_df(24)
    funding_raw = pd.DataFrame(
        {
            "funding_time": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T08:00:00Z", "2024-01-01T16:00:00Z"], utc=True
            ),
            "funding_rate": [0.001, 0.002, 0.003],
        }
    )
    funding_1h = process_funding_to_1h(funding_raw, "2024-01-01", "2024-01-02")
    merged = merge_ohlcv_with_funding(ohlcv, funding_1h)

    assert merged.loc[0, "funding_rate"] == 0.001
    assert merged.loc[7, "funding_rate"] == 0.001
    assert merged.loc[8, "funding_rate"] == 0.002
    assert merged.loc[16, "funding_rate"] == 0.003
