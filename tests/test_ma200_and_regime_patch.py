import pandas as pd

from bot.backtest.engine import BacktestEngine
from bot.backtest.metrics import compute_metrics
from bot.cli import _build_summary
from bot.features.builder import build_features
from bot.regime.detector import RegimeDetector
from bot.runs.run_manager import RunManager
from bot.strategy.breakout_atr import BreakoutATRStrategy, Signal
from bot.utils.config import RegimeSettings, StrategyBreakoutSettings, load_settings


def _base_df(n: int = 320) -> pd.DataFrame:
    close = [100 + i * 0.25 for i in range(n)]
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
    out = build_features(df)
    out["regime"] = "TREND"
    out["rel_volume_24"] = 2.0
    return out


def test_ma200_filter_directional() -> None:
    i = 280

    above = _base_df()
    above.loc[above.index[i], "ma_200"] = above.loc[above.index[i], "close"] - 10.0
    strategy = BreakoutATRStrategy(StrategyBreakoutSettings(mode="ema", use_ma200_filter=True))
    long_signal = strategy._ma200_filter_reason(above.iloc[i], signal=Signal(side="LONG", reason="x"))
    short_signal = strategy._ma200_filter_reason(above.iloc[i], signal=Signal(side="SHORT", reason="x"))
    assert long_signal is None
    assert short_signal == "ma200"

    below = _base_df()
    below.loc[below.index[i], "ma_200"] = below.loc[below.index[i], "close"] + 10.0
    long_signal = strategy._ma200_filter_reason(below.iloc[i], signal=Signal(side="LONG", reason="x"))
    short_signal = strategy._ma200_filter_reason(below.iloc[i], signal=Signal(side="SHORT", reason="x"))
    assert long_signal == "ma200"
    assert short_signal is None


def test_trend_threshold_reduces_trend_share() -> None:
    df = _base_df(400)
    detector_low = RegimeDetector(RegimeSettings(adx_trend_threshold=20))
    detector_high = RegimeDetector(RegimeSettings(adx_trend_threshold=28))

    trend_low = int((detector_low.apply(df) == "TREND").sum())
    trend_high = int((detector_high.apply(df) == "TREND").sum())
    assert trend_high <= trend_low


def test_backtest_still_runs_and_writes_artifacts(tmp_path) -> None:
    cfg = load_settings()
    cfg.strategy_breakout.mode = "ema"

    df = _base_df(360)
    df["regime"] = RegimeDetector(cfg.regime).apply(df)

    engine = BacktestEngine(cfg)
    trades, equity = engine.run(df)
    summary = _build_summary(trades, equity, compute_metrics(trades, equity), engine.last_run_diagnostics)

    manager = RunManager(tmp_path)
    ctx = manager.build_context(cfg, run_name="smoke")
    manager.init_run(ctx, settings=cfg, data_path="data/mock.parquet", config_path="config/settings.yaml")
    manager.persist_outputs(
        ctx,
        summary=summary,
        trades=trades,
        equity=equity,
        regime_stats={"TREND": {}},
        direction_stats={"LONG": {}, "SHORT": {}},
    )

    assert (ctx.run_dir / "summary.json").exists()
    assert (ctx.run_dir / "trades.csv").exists()
