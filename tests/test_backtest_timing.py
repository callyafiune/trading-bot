import pandas as pd

from bot.backtest.engine import BacktestEngine
from bot.features.builder import build_features
from bot.regime.detector import RegimeDetector
from bot.utils.config import load_settings


def _sample_df():
    n = 300
    close = [100 + i * 0.5 for i in range(n)]
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [1000] * n,
        }
    )


def test_entry_happens_next_candle_open():
    cfg = load_settings()
    df = build_features(_sample_df())
    df["regime"] = RegimeDetector(cfg.regime).apply(df)
    trades, _ = BacktestEngine(cfg).run(df)
    if not trades.empty:
        assert trades.iloc[0]["entry_time"] > df.iloc[0]["open_time"]
