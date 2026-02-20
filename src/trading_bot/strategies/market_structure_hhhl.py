from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot.config import MarketStructureConfig


@dataclass
class SwingPoint:
    kind: str  # HIGH or LOW
    pivot_idx: int
    confirm_idx: int
    price: float
    label: str  # HH HL LH LL SH SL


def _rolling_atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()


def detect_confirmed_swings(df: pd.DataFrame, cfg: MarketStructureConfig) -> tuple[list[SwingPoint], pd.DataFrame]:
    left = int(cfg.pivot_left)
    right = int(cfg.pivot_right)
    src_high = df["close"] if cfg.use_close_for_swings else df["high"]
    src_low = df["close"] if cfg.use_close_for_swings else df["low"]

    n = len(df)
    swings: list[SwingPoint] = []
    out = pd.DataFrame(index=df.index)
    out["swing_high_confirmed"] = False
    out["swing_low_confirmed"] = False
    out["swing_high_price"] = np.nan
    out["swing_low_price"] = np.nan
    out["swing_high_label"] = ""
    out["swing_low_label"] = ""

    last_high_price: float | None = None
    last_low_price: float | None = None

    for i in range(left, n - right):
        c_high = float(src_high.iat[i])
        c_low = float(src_low.iat[i])
        prev_highs = src_high.iloc[i - left : i]
        next_highs = src_high.iloc[i + 1 : i + 1 + right]
        prev_lows = src_low.iloc[i - left : i]
        next_lows = src_low.iloc[i + 1 : i + 1 + right]

        is_high = bool((c_high > prev_highs).all() and (c_high > next_highs).all())
        is_low = bool((c_low < prev_lows).all() and (c_low < next_lows).all())
        confirm_idx = i + right

        if is_high:
            if last_high_price is None:
                label = "SH"
            else:
                label = "HH" if c_high > last_high_price else "LH"
            last_high_price = c_high
            swings.append(SwingPoint("HIGH", i, confirm_idx, c_high, label))
            out.at[confirm_idx, "swing_high_confirmed"] = True
            out.at[confirm_idx, "swing_high_price"] = c_high
            out.at[confirm_idx, "swing_high_label"] = label

        if is_low:
            if last_low_price is None:
                label = "SL"
            else:
                label = "HL" if c_low > last_low_price else "LL"
            last_low_price = c_low
            swings.append(SwingPoint("LOW", i, confirm_idx, c_low, label))
            out.at[confirm_idx, "swing_low_confirmed"] = True
            out.at[confirm_idx, "swing_low_price"] = c_low
            out.at[confirm_idx, "swing_low_label"] = label

    swings.sort(key=lambda x: x.confirm_idx)
    return swings, out


def prepare_structure_features(df: pd.DataFrame, cfg: MarketStructureConfig) -> tuple[pd.DataFrame, list[SwingPoint]]:
    swings, swing_df = detect_confirmed_swings(df, cfg)
    out = df.copy()
    out = pd.concat([out, swing_df], axis=1)

    out["ma99"] = out["close"].rolling(99, min_periods=99).mean()
    trend_len = int(cfg.trend_ma_length)
    out[f"ma_{trend_len}"] = out["close"].rolling(trend_len, min_periods=trend_len).mean()
    out["atr"] = _rolling_atr(out, int(cfg.atr_length))
    out["atr_pct"] = out["atr"] / out["close"]

    out["last_swing_high_price"] = out["swing_high_price"].ffill()
    out["last_swing_low_price"] = out["swing_low_price"].ffill()
    out["last_swing_high_label"] = out["swing_high_label"].replace("", np.nan).ffill().fillna("")
    out["last_swing_low_label"] = out["swing_low_label"].replace("", np.nan).ffill().fillna("")

    return out, swings


def generate_structure_signals(df: pd.DataFrame, cfg: MarketStructureConfig) -> pd.Series:
    sig = pd.Series(0, index=df.index, dtype=int)

    last_ll_idx: int | None = None
    last_hl_idx: int | None = None
    last_hh_idx: int | None = None
    last_lh_idx: int | None = None

    for i, row in df.iterrows():
        low_label = str(row.get("swing_low_label", ""))
        high_label = str(row.get("swing_high_label", ""))

        if low_label == "LL":
            last_ll_idx = int(i)
        elif low_label == "HL":
            last_hl_idx = int(i)

        if high_label == "HH":
            last_hh_idx = int(i)
        elif high_label == "LH":
            last_lh_idx = int(i)

        long_ok = high_label == "HH" and last_ll_idx is not None and last_hl_idx is not None and last_hl_idx > last_ll_idx
        short_ok = low_label == "LL" and last_hh_idx is not None and last_lh_idx is not None and last_lh_idx > last_hh_idx

        if cfg.filter_ma99:
            if long_ok and not bool(row.get("close", np.nan) > row.get("ma99", np.nan)):
                long_ok = False
            if short_ok and not bool(row.get("close", np.nan) < row.get("ma99", np.nan)):
                short_ok = False

        if cfg.filter_trend_ma:
            ma_col = f"ma_{int(cfg.trend_ma_length)}"
            if long_ok and not bool(row.get("close", np.nan) > row.get(ma_col, np.nan)):
                long_ok = False
            if short_ok and not bool(row.get("close", np.nan) < row.get(ma_col, np.nan)):
                short_ok = False

        if cfg.filter_atr_vol:
            atr_pct = float(row.get("atr_pct", np.nan))
            if np.isnan(atr_pct) or atr_pct < float(cfg.atr_pct_min):
                long_ok = False
                short_ok = False

        if cfg.filter_session and cfg.allowed_hours_utc:
            ts = row.get("open_time")
            hour = int(getattr(ts, "hour", -1))
            if hour not in set(cfg.allowed_hours_utc):
                long_ok = False
                short_ok = False

        if long_ok and not short_ok:
            sig.iat[i] = 1
        elif short_ok and not long_ok:
            sig.iat[i] = -1

    return sig


def pick_mode_a_target(df: pd.DataFrame, i: int, side: int, lookback: int) -> float | None:
    start = max(0, i - lookback)
    hist = df.iloc[start:i]
    entry = float(df["close"].iat[i])

    if side > 0:
        highs = hist.loc[hist["swing_high_confirmed"].fillna(False), "swing_high_price"].dropna()
        highs = highs[highs > entry]
        if highs.empty:
            return None
        nearest = (highs - entry).abs().sort_values()
        return float(highs.loc[nearest.index[0]])

    lows = hist.loc[hist["swing_low_confirmed"].fillna(False), "swing_low_price"].dropna()
    lows = lows[lows < entry]
    if lows.empty:
        return None
    nearest = (lows - entry).abs().sort_values()
    return float(lows.loc[nearest.index[0]])


def structure_trailing_exit(side: int, row: pd.Series) -> bool:
    if side > 0:
        return str(row.get("swing_high_label", "")) == "LH" and float(row.get("close", np.nan)) < float(
            row.get("last_swing_low_price", np.nan)
        )
    return str(row.get("swing_low_label", "")) == "HL" and float(row.get("close", np.nan)) > float(
        row.get("last_swing_high_price", np.nan)
    )
