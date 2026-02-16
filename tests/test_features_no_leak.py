import numpy as np
import pandas as pd

from bot.features.builder import build_features


def test_log_return_1_no_future_leak():
    df = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=30, freq="H", tz="UTC"),
            "open": range(30),
            "high": [x + 1 for x in range(30)],
            "low": [max(x - 1, 0) for x in range(30)],
            "close": [100 + x for x in range(30)],
            "volume": [10] * 30,
        }
    )
    feat = build_features(df)
    expected = (feat["close"].iloc[10] / feat["close"].iloc[9]).__float__()
    assert abs(feat["log_return_1"].iloc[10] - np.log(expected)) < 1e-9
