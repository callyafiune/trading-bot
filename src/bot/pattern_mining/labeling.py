from __future__ import annotations

import numpy as np
import pandas as pd


def _future_window_stats(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(closes)
    future_close = np.full(n, np.nan, dtype=float)
    future_max_high = np.full(n, np.nan, dtype=float)
    future_min_low = np.full(n, np.nan, dtype=float)

    for i in range(n):
        end_idx = i + horizon
        if end_idx >= n:
            continue
        window_slice = slice(i + 1, end_idx + 1)
        future_close[i] = closes[end_idx]
        future_max_high[i] = np.nanmax(highs[window_slice])
        future_min_low[i] = np.nanmin(lows[window_slice])

    return future_close, future_max_high, future_min_low


def add_labels(
    df: pd.DataFrame,
    horizon: int = 3,
    up_threshold: float = 0.005,
    down_threshold: float = -0.005,
) -> pd.DataFrame:
    out = df.copy()
    closes = out["close"].to_numpy(dtype=float)
    highs = out["high"].to_numpy(dtype=float)
    lows = out["low"].to_numpy(dtype=float)

    future_close, future_max_high, future_min_low = _future_window_stats(highs, lows, closes, horizon)

    future_return = (future_close / closes) - 1.0
    out[f"future_return_{horizon}h"] = future_return
    out["y_up"] = future_return >= up_threshold
    out["y_down"] = future_return <= down_threshold

    up_touch_level = closes * (1.0 + up_threshold)
    down_touch_level = closes * (1.0 + down_threshold)
    out["touch_up"] = future_max_high >= up_touch_level
    out["touch_down"] = future_min_low <= down_touch_level

    out["max_drawdown_future"] = (future_min_low / closes) - 1.0
    out["max_runup_future"] = (future_max_high / closes) - 1.0
    out[f"max_ddown_{horizon}h"] = out["max_drawdown_future"]
    out[f"max_runup_{horizon}h"] = out["max_runup_future"]

    return out
