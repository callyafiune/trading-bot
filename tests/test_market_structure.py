import pandas as pd

from bot.features.builder import build_features
from bot.strategy.breakout_atr import BreakoutATRStrategy
from bot.utils.config import MarketStructureSettings, StrategyBreakoutSettings, StrategyRouterSettings


def _ohlcv_from_close(close: list[float], high: list[float] | None = None, low: list[float] | None = None) -> pd.DataFrame:
    high = high or [c + 0.5 for c in close]
    low = low or [c - 0.5 for c in close]
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=len(close), freq="h", tz="UTC"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1000.0] * len(close),
        }
    )


def test_swings_no_lookahead() -> None:
    close = [10, 11, 15, 12, 11, 12, 13, 12, 11]
    high = [10, 11, 15, 12, 11, 12, 13, 12, 11]
    low = [9, 8, 6, 8, 9, 8, 7, 8, 9]
    ms = MarketStructureSettings(enabled=True, left_bars=2, right_bars=2)

    df = build_features(_ohlcv_from_close(close, high=high, low=low), market_structure_settings=ms)

    assert pd.isna(df.loc[0, "swing_high_price"])
    assert pd.isna(df.loc[3, "swing_high_price"])
    assert df.loc[4, "swing_high_price"] == 15
    assert int(df.loc[4, "swing_high_idx"]) == 2


def test_hh_hl_lh_ll_sequence() -> None:
    high = [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.8, 3.2, 4.0, 3.0, 3.6, 2.8]
    low = [0.5, 1.5, 1.0, 2.0, 1.8, 2.5, 2.0, 2.2, 1.6, 2.1, 1.2, 1.8, 0.9, 1.5, 0.8]
    close = [(h + l) / 2.0 for h, l in zip(high, low)]
    ms = MarketStructureSettings(enabled=True, left_bars=1, right_bars=1)

    df = build_features(_ohlcv_from_close(close, high=high, low=low), market_structure_settings=ms)

    high_event_types = df.loc[df["swing_high_price"].notna(), "ms_last_high_type"].dropna().tolist()
    low_event_types = df.loc[df["swing_low_price"].notna(), "ms_last_low_type"].dropna().tolist()

    assert "HH" in high_event_types
    assert "LH" in high_event_types
    assert "HL" in low_event_types
    assert "LL" in low_event_types


def test_msb_detection() -> None:
    high = [11.0, 15.0, 13.0, 14.0, 12.0, 13.0, 12.0, 15.5, 14.0, 13.0, 12.0, 11.0]
    low = [10.0, 12.0, 8.0, 11.0, 9.0, 10.0, 9.5, 13.0, 8.5, 8.8, 7.5, 7.8]
    close = [10.5, 14.0, 9.0, 13.0, 10.0, 12.0, 10.0, 15.0, 9.2, 8.6, 7.6, 8.0]

    ms = MarketStructureSettings(enabled=True, left_bars=1, right_bars=1)
    ms.msb.enabled = True
    ms.msb.persist_bars = 3

    df = build_features(_ohlcv_from_close(close, high=high, low=low), market_structure_settings=ms)

    assert bool(df["msb_bull"].any())
    assert bool(df["msb_bear"].any())
    assert float(df.loc[df["msb_bull"], "msb_level"].iloc[0]) == 13.0


def test_gate_blocks_trades() -> None:
    close = [100.0 + i for i in range(40)]
    df = build_features(_ohlcv_from_close(close))
    i = 20
    df["regime"] = "BULL_TREND"
    df["ms_structure_state"] = "NEUTRAL"
    df["msb_bull"] = False
    df["msb_bull_active"] = False
    df["msb_bear"] = False
    df["msb_bear_active"] = False

    df.loc[df.index[i - 1], ["ema_12", "ema_26"]] = [99.0, 100.0]
    df.loc[df.index[i], ["ema_12", "ema_26"]] = [101.0, 100.0]

    ms = MarketStructureSettings(enabled=True)
    ms.gate.enabled = True
    ms.gate.mode = "structure_trend"
    ms.gate.block_in_neutral = True

    strategy = BreakoutATRStrategy(
        settings=StrategyBreakoutSettings(mode="ema", use_ma200_filter=False),
        router_settings=StrategyRouterSettings(),
        market_structure=ms,
    )
    signal, reason = strategy.signal_decision(df, i)

    assert signal is None
    assert reason == "ms_gate_long"
