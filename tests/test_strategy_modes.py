import pandas as pd

from bot.features.builder import build_features
from bot.strategy.breakout_atr import BreakoutATRStrategy
from bot.utils.config import StrategyBreakoutSettings


def _base_df() -> pd.DataFrame:
    n = 260
    close = [100 + i * 0.3 for i in range(n)]
    df = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [2000.0] * n,
        }
    )
    df = build_features(df)
    df["regime"] = "TREND"
    df["rel_volume_24"] = 2.0
    return df


def test_all_modes_respect_regime_detector() -> None:
    df = _base_df()
    i = 220
    df.loc[df.index[i], "regime"] = "RANGE"

    for mode in ["breakout", "baseline", "ema", "ema_macd", "ml_gate"]:
        cfg = StrategyBreakoutSettings(mode=mode, use_ma200_filter=False)
        strategy = BreakoutATRStrategy(cfg)
        sig, reason = strategy.signal_decision(df, i)
        assert sig is None
        assert reason == "regime"


def test_ema_crossover_matches_classic_signal() -> None:
    df = _base_df()
    i = 30
    df.loc[df.index[i - 1], ["ema_12", "ema_26"]] = [99.0, 100.0]
    df.loc[df.index[i], ["ema_12", "ema_26"]] = [101.0, 100.0]

    strategy = BreakoutATRStrategy(StrategyBreakoutSettings(mode="ema", use_ma200_filter=False))
    sig, reason = strategy.signal_decision(df, i)
    assert reason is None
    assert sig is not None
    assert sig.side == "LONG"
    assert sig.reason == "ema_cross_up"


def test_ml_gate_blocks_when_probability_below_threshold() -> None:
    df = _base_df()
    i = 36
    # cria um crossover long válido + MACD/volume
    df.loc[df.index[i - 1], ["ema_12", "ema_26"]] = [99.0, 100.0]
    df.loc[df.index[i], ["ema_12", "ema_26"]] = [101.0, 100.0]
    df.loc[df.index[i], ["macd_line", "macd_signal", "rel_volume_24"]] = [2.0, 1.0, 1.2]
    df["ml_prob"] = 0.51
    df["ml_threshold"] = 0.55

    strategy = BreakoutATRStrategy(StrategyBreakoutSettings(mode="ml_gate", ml_prob_threshold=0.55, use_ma200_filter=False))
    sig, reason = strategy.signal_decision(df, i)
    assert sig is None
    assert reason == "ml_gate"


def test_baseline_mode_generates_signals() -> None:
    df = _base_df()
    strategy = BreakoutATRStrategy(
        StrategyBreakoutSettings(mode="baseline", breakout_lookback_N=20, use_ma200_filter=False, use_rel_volume_filter=False)
    )

    signals_total = 0
    blocked_mode = 0
    for i in range(1, len(df) - 1):
        decision = strategy.evaluate_signal(df, i)
        if decision.raw_signal is not None:
            signals_total += 1
        if decision.blocked_reason in ("macd_gate", "ml_gate", "unsupported_mode"):
            blocked_mode += 1

    assert signals_total > 0
    assert blocked_mode == 0
