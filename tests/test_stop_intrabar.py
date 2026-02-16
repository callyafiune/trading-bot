import pandas as pd

from bot.backtest.engine import BacktestEngine
from bot.utils.config import load_settings


def test_stop_intrabar_priority():
    cfg = load_settings()
    df = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=5, freq="H", tz="UTC"),
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 150, 104, 105],
            "low": [99, 100, 50, 102, 103],
            "close": [100, 101, 103, 104, 104],
            "volume": [2000] * 5,
            "atr_14": [1] * 5,
            "regime": ["TREND"] * 5,
            "rel_volume_24": [2] * 5,
        }
    )
    trades, _ = BacktestEngine(cfg).run(df)
    assert isinstance(trades, pd.DataFrame)
