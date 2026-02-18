import numpy as np
import pandas as pd

from bot.features.builder import build_features


def test_log_return_1_no_future_leak():
    df = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
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


def test_no_lookahead_ma200():
    n = 260
    close = pd.Series([100 + i for i in range(n)], dtype=float)
    df = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [10] * n,
        }
    )
    feat = build_features(df)
    expected = close.rolling(200, min_periods=200).mean()
    pd.testing.assert_series_equal(feat["ma_200"], expected, check_names=False)


def test_no_lookahead_features():
    n = 300
    close = pd.Series([100 + i for i in range(n)], dtype=float)
    df = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [10] * n,
        }
    )
    feat = build_features(df)

    expected_slope24 = (feat["ma200"] - feat["ma200"].shift(24)) / feat["ma200"].shift(24)
    expected_vol24 = np.log(feat["close"]).diff().rolling(24, min_periods=24).std()
    expected_vol168 = np.log(feat["close"]).diff().rolling(168, min_periods=168).std()

    pd.testing.assert_series_equal(feat["slope_ma200"], expected_slope24, check_names=False)
    pd.testing.assert_series_equal(feat["rolling_vol_24h"], expected_vol24, check_names=False)
    pd.testing.assert_series_equal(feat["rolling_vol_168h"], expected_vol168, check_names=False)
