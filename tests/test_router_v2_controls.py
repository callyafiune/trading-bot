import pandas as pd

from bot.backtest.engine import BacktestEngine
from bot.features.builder import build_features
from bot.strategy.breakout_atr import BreakoutATRStrategy
from bot.utils.config import Settings, StrategyBreakoutSettings, StrategyRouterSettings


def _base_ohlcv(n: int = 320) -> pd.DataFrame:
    close = [100.0 + (i * 0.1) for i in range(n)]
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": close,
            "high": [c + 0.5 for c in close],
            "low": [c - 0.5 for c in close],
            "close": close,
            "volume": [2000.0] * n,
        }
    )


def test_range_is_flat() -> None:
    df = _base_ohlcv()
    for idx in range(250, 280):
        df.loc[df.index[idx], ["close", "high", "low"]] = [500.0 + idx, 500.0 + idx, 500.0 + idx]

    df = build_features(df)
    df["regime"] = "BULL_RANGE"

    cfg = Settings(start_date="2024-01-01", end_date="2024-02-01")
    cfg.strategy_breakout = StrategyBreakoutSettings(mode="breakout", breakout_lookback_N=20, use_ma200_filter=False, time_stop_hours=1)
    cfg.strategy_router = StrategyRouterSettings(enable_range=False)

    trades, _ = BacktestEngine(cfg).run(df)
    assert trades.empty


def test_cooldown_blocks_reentry() -> None:
    df = _base_ohlcv()
    for idx in range(250, 280):
        df.loc[df.index[idx], ["close", "high", "low"]] = [800.0 + idx, 800.0 + idx, 800.0 + idx]

    df = build_features(df)
    df["regime"] = "BULL_TREND"
    df["ema50"] = df["close"] + 10.0
    df["ema200"] = df["close"] - 10.0
    df["slope_ema200_pct"] = 0.01

    cfg = Settings(start_date="2024-01-01", end_date="2024-02-01")
    cfg.strategy_breakout = StrategyBreakoutSettings(mode="breakout", breakout_lookback_N=20, use_ma200_filter=False, time_stop_hours=1)
    cfg.strategy_router = StrategyRouterSettings(cooldown_bars_after_exit=6)

    engine = BacktestEngine(cfg)
    trades, _ = engine.run(df)

    assert len(trades) >= 1
    if len(trades) > 1:
        entry_idx = [df.index[df["open_time"] == ts][0] for ts in trades["entry_time"]]
        for prev_idx, next_idx in zip(entry_idx, entry_idx[1:]):
            assert (next_idx - prev_idx) > 6
    assert int(engine.last_run_diagnostics.get("blocked_cooldown", 0)) > 0


def test_bull_trend_filters() -> None:
    n = 220
    close = [100.0] * n
    i = 180
    close[i] = 140.0

    df = pd.DataFrame(
        {
            "high": close,
            "low": close,
            "close": close,
            "regime": ["BULL_TREND"] * n,
            "rel_volume_24": [2.0] * n,
            "ma_200": [90.0] * n,
            "ema50": [80.0] * n,
            "ema200": [100.0] * n,
            "slope_ema200_pct": [0.01] * n,
        }
    )

    strategy = BreakoutATRStrategy(
        StrategyBreakoutSettings(mode="breakout", breakout_lookback_N=20, use_ma200_filter=False, use_rel_volume_filter=False),
        StrategyRouterSettings(),
    )

    sig, reason = strategy.signal_decision(df, i)

    assert sig is None
    assert reason == "blocked_trend_up_filter"
