from __future__ import annotations

import numpy as np
import pandas as pd

from bot.utils.config import FeatureSettings


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    return _true_range(df).rolling(period, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(df)
    atr = tr.rolling(period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    return dx.rolling(period, min_periods=period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame, settings: FeatureSettings | None = None) -> pd.DataFrame:
    settings = settings or FeatureSettings()
    out = df.copy()
    log_price = np.log(out["close"])
    for n in [1, 2, 4, 8, 24]:
        out[f"log_return_{n}"] = log_price.diff(n)

    ret1 = out["close"].pct_change()
    for n in [6, 24, 72]:
        out[f"realized_vol_{n}"] = ret1.rolling(n, min_periods=n).std()

    out["atr14"] = _atr(out, 14)
    out["atr_14"] = out["atr14"]
    out["atr_pct"] = out["atr14"] / out["close"]

    out["ema_12"] = _ema(out["close"], 12)
    out["ema_26"] = _ema(out["close"], 26)
    out["macd_line"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = _ema(out["macd_line"], 9)
    out["rsi_14"] = _rsi(out["close"], 14)

    for n in [20, 50, 200]:
        out[f"ma_{n}"] = out["close"].rolling(n, min_periods=n).mean()
        out[f"dist_ma_{n}_pct"] = (out["close"] - out[f"ma_{n}"]) / out[f"ma_{n}"]
    out["ma200"] = out["ma_200"]

    out["adx14"] = _adx(out, 14)
    out["adx_14"] = out["adx14"]

    out["ema50"] = _ema(out["close"], settings.ema_fast)
    out["ema200"] = _ema(out["close"], settings.ema_slow)
    out["slope_ema200"] = (out["ema200"] / out["ema200"].shift(settings.slope_window)) - 1.0
    out["slope_ema200_pct"] = out["slope_ema200"]
    out["slope_ma200"] = (out["ma200"] - out["ma200"].shift(24)) / out["ma200"].shift(24)
    out["slope_ma200_72h"] = (out["ma200"] - out["ma200"].shift(72)) / out["ma200"].shift(72)

    log_ret_1 = np.log(out["close"]).diff()
    vol_roll = log_ret_1.rolling(settings.vol_window, min_periods=settings.vol_window).std()
    if settings.annualize_vol:
        vol_roll = vol_roll * np.sqrt(24 * 365)
    out["vol_roll"] = vol_roll
    out["rolling_vol_24h"] = log_ret_1.rolling(24, min_periods=24).std()
    out["rolling_vol_168h"] = log_ret_1.rolling(168, min_periods=168).std()

    bb_mid = out["close"].rolling(20, min_periods=20).mean()
    bb_std = out["close"].rolling(20, min_periods=20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_dn = bb_mid - 2 * bb_std
    out["bb_width_20"] = (bb_up - bb_dn) / bb_mid

    out["rel_volume_24"] = out["volume"] / out["volume"].rolling(24, min_periods=24).mean()
    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["slope_24"] = out["close"].pct_change(24)

    if "funding_rate" in out.columns:
        window = 168
        mean = out["funding_rate"].rolling(window, min_periods=window).mean()
        std = out["funding_rate"].rolling(window, min_periods=window).std(ddof=0)
        out["funding_z"] = (out["funding_rate"] - mean) / std.replace(0.0, np.nan)

    return out
