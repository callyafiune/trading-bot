from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from trading_bot.regime_shift_config import RegimeShiftConfig

Regime = Literal["unknown", "bullish", "bearish"]


@dataclass
class State:
    major_high: float | None = None
    major_low: float | None = None
    regime: Regime = "unknown"


def compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()


def _min_move(row: pd.Series, cfg: RegimeShiftConfig) -> float:
    if cfg.significance_mode == "atr":
        atr = float(row.get("atr", np.nan))
        if np.isnan(atr):
            return np.inf
        return atr * float(cfg.atr_k)
    return float(row["close"]) * float(cfg.percent_p)


def _buffer(row: pd.Series, cfg: RegimeShiftConfig) -> float:
    if cfg.break_confirm_mode == "close_only":
        return 0.0
    if cfg.buffer_mode == "atr":
        atr = float(row.get("atr", np.nan))
        if np.isnan(atr):
            return np.inf
        return atr * float(cfg.atr_buffer_k)
    return float(row["close"]) * float(cfg.percent_buffer_p)


def _break_up(df: pd.DataFrame, i: int, major_high: float, buffer: float, cfg: RegimeShiftConfig) -> bool:
    close_i = float(df["close"].iat[i])
    level = major_high + buffer
    if cfg.break_confirm_mode in ("close_only", "close_plus_buffer"):
        return close_i > level

    n = max(2, int(cfg.n_closes_confirm))
    if i - n + 1 < 0:
        return False
    closes = df["close"].iloc[i - n + 1 : i + 1]
    return bool((closes > level).all())


def _break_down(df: pd.DataFrame, i: int, major_low: float, buffer: float, cfg: RegimeShiftConfig) -> bool:
    close_i = float(df["close"].iat[i])
    level = major_low - buffer
    if cfg.break_confirm_mode in ("close_only", "close_plus_buffer"):
        return close_i < level

    n = max(2, int(cfg.n_closes_confirm))
    if i - n + 1 < 0:
        return False
    closes = df["close"].iloc[i - n + 1 : i + 1]
    return bool((closes < level).all())


def detect_regime_shift_levels(df: pd.DataFrame, cfg: RegimeShiftConfig) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = compute_atr(out, int(cfg.atr_len))
    out["ma99"] = out["close"].rolling(99, min_periods=99).mean()

    out["major_high"] = np.nan
    out["major_low"] = np.nan
    out["regime"] = "unknown"
    out["bullish_break"] = False
    out["bearish_break"] = False
    out["bullish_shift"] = False
    out["bearish_shift"] = False

    s = State()
    lb = int(cfg.lookback_init)

    for i in range(len(out)):
        row = out.iloc[i]
        if i < lb:
            out.at[i, "regime"] = s.regime
            continue

        if s.major_high is None or s.major_low is None:
            hist = out.iloc[i - lb : i]
            s.major_high = float(hist["high"].max())
            s.major_low = float(hist["low"].min())

        mh_pre = float(s.major_high)
        ml_pre = float(s.major_low)
        m_move = _min_move(row, cfg)
        buf = _buffer(row, cfg)

        bull_break = _break_up(out, i, mh_pre, buf, cfg)
        bear_break = _break_down(out, i, ml_pre, buf, cfg)

        bull_shift = bull_break and s.regime != "bullish"
        bear_shift = bear_break and s.regime != "bearish"

        if bull_shift:
            s.regime = "bullish"
        elif bear_shift:
            s.regime = "bearish"

        h = float(row["high"])
        l = float(row["low"])

        if s.regime == "bullish":
            if h > s.major_high + m_move:
                s.major_high = h
            if (s.major_high - l) >= m_move:
                s.major_low = max(float(s.major_low), l)
        elif s.regime == "bearish":
            if l < s.major_low - m_move:
                s.major_low = l
            if (h - s.major_low) >= m_move:
                s.major_high = min(float(s.major_high), h)

        out.at[i, "major_high"] = float(s.major_high)
        out.at[i, "major_low"] = float(s.major_low)
        out.at[i, "regime"] = s.regime
        out.at[i, "bullish_break"] = bool(bull_break)
        out.at[i, "bearish_break"] = bool(bear_break)
        out.at[i, "bullish_shift"] = bool(bull_shift)
        out.at[i, "bearish_shift"] = bool(bear_shift)

    return out
