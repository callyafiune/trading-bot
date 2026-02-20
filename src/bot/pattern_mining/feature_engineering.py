from __future__ import annotations

import numpy as np
import pandas as pd


PATTERN_FEATURE_COLUMNS = [
    "range",
    "body",
    "lower_wick_ratio",
    "upper_wick_ratio",
    "close_position_in_range",
    "log_return",
    "rolling_volatility",
    "ma7",
    "ma25",
    "ma99",
    "dist_to_ma7",
    "dist_to_ma25",
    "dist_to_ma99",
    "slope_ma99",
    "var_oi_1",
    "var_oi_3",
    "var_cvd_1",
    "var_cvd_3",
    "volume_zscore",
]


def regime_id_from_flags(price_above_ma99: bool, ma99_slope_up: bool, vol_high: bool) -> str:
    return f"pa{int(price_above_ma99)}_sl{int(ma99_slope_up)}_vh{int(vol_high)}"


def build_regime_flags(
    df: pd.DataFrame,
    *,
    vol_quantile: float = 0.75,
    vol_history_min: int = 50,
) -> pd.DataFrame:
    out = df.copy()
    if "ma99" not in out.columns or "slope_ma99" not in out.columns:
        out = build_pattern_features(out)

    rolling_vol = out.get("rolling_volatility")
    if rolling_vol is None:
        rolling_vol = np.log(out["close"]).diff(1).rolling(20, min_periods=20).std()

    # Historical-only threshold: quantile computed from past values up to t-1.
    vol_threshold = rolling_vol.shift(1).expanding(min_periods=vol_history_min).quantile(vol_quantile)

    out["regime_price_above_ma99"] = (out["close"] > out["ma99"]).fillna(False)
    out["regime_ma99_slope_up"] = (out["slope_ma99"] > 0).fillna(False)
    out["regime_vol_high"] = ((rolling_vol > vol_threshold) & vol_threshold.notna()).fillna(False)

    out["regime_id"] = [
        regime_id_from_flags(pa, sl, vh)
        for pa, sl, vh in zip(
            out["regime_price_above_ma99"].astype(bool),
            out["regime_ma99_slope_up"].astype(bool),
            out["regime_vol_high"].astype(bool),
        )
    ]
    return out


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def build_pattern_features(
    df: pd.DataFrame,
    volatility_window: int = 20,
    volume_z_window: int = 30,
) -> pd.DataFrame:
    out = df.copy()

    out["range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()

    lower_wick = np.minimum(out["open"], out["close"]) - out["low"]
    upper_wick = out["high"] - np.maximum(out["open"], out["close"])
    out["lower_wick_ratio"] = _safe_div(lower_wick, out["range"])
    out["upper_wick_ratio"] = _safe_div(upper_wick, out["range"])
    out["close_position_in_range"] = _safe_div(out["close"] - out["low"], out["range"])

    out["log_return"] = np.log(out["close"]).diff(1)
    out["rolling_volatility"] = out["log_return"].rolling(volatility_window, min_periods=volatility_window).std()

    out["ma7"] = out["close"].rolling(7, min_periods=7).mean()
    out["ma25"] = out["close"].rolling(25, min_periods=25).mean()
    out["ma99"] = out["close"].rolling(99, min_periods=99).mean()

    out["dist_to_ma7"] = _safe_div(out["close"] - out["ma7"], out["ma7"])
    out["dist_to_ma25"] = _safe_div(out["close"] - out["ma25"], out["ma25"])
    out["dist_to_ma99"] = _safe_div(out["close"] - out["ma99"], out["ma99"])
    out["slope_ma99"] = _safe_div(out["ma99"] - out["ma99"].shift(1), out["ma99"].shift(1))

    if "oi" in out.columns:
        out["var_oi_1"] = out["oi"].pct_change(1)
        out["var_oi_3"] = out["oi"].pct_change(3)
    else:
        out["var_oi_1"] = np.nan
        out["var_oi_3"] = np.nan

    if "cvd" in out.columns:
        out["var_cvd_1"] = out["cvd"].pct_change(1)
        out["var_cvd_3"] = out["cvd"].pct_change(3)
    else:
        out["var_cvd_1"] = np.nan
        out["var_cvd_3"] = np.nan

    vol_mean = out["volume"].rolling(volume_z_window, min_periods=volume_z_window).mean()
    vol_std = out["volume"].rolling(volume_z_window, min_periods=volume_z_window).std(ddof=0)
    out["volume_zscore"] = _safe_div(out["volume"] - vol_mean, vol_std)

    return out
