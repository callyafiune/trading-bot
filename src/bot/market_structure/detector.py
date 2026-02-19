from __future__ import annotations

import numpy as np
import pandas as pd

from bot.utils.config import MarketStructureSettings


def _confirm_swings(
    values: pd.Series,
    left_bars: int,
    right_bars: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = left_bars + right_bars + 1
    if window <= 1:
        idx = np.arange(len(values), dtype=int)
        return idx, idx, values.to_numpy(dtype=float)

    centered = values.rolling(window=window, min_periods=window, center=True)
    if mode == "max":
        extremum = centered.max()
        is_swing = values.eq(extremum)
    else:
        extremum = centered.min()
        is_swing = values.eq(extremum)

    pivot_idx = np.flatnonzero(is_swing.to_numpy(dtype=bool))
    if pivot_idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    confirm_idx = pivot_idx + right_bars
    valid = confirm_idx < len(values)
    pivot_idx = pivot_idx[valid]
    confirm_idx = confirm_idx[valid]
    prices = values.to_numpy(dtype=float)[pivot_idx]
    return pivot_idx.astype(int), confirm_idx.astype(int), prices.astype(float)


def add_market_structure_features(
    df: pd.DataFrame,
    settings: MarketStructureSettings | None,
) -> pd.DataFrame:
    out = df.copy()
    open_time = pd.to_datetime(out["open_time"], utc=True)

    cols_defaults: dict[str, object] = {
        "swing_high_price": np.nan,
        "swing_low_price": np.nan,
        "swing_high_confirmed_at": pd.Series([None] * len(out), index=out.index, dtype="object"),
        "swing_low_confirmed_at": pd.Series([None] * len(out), index=out.index, dtype="object"),
        "swing_high_idx": np.nan,
        "swing_low_idx": np.nan,
        "ms_last_swing_high": np.nan,
        "ms_last_swing_low": np.nan,
        "ms_last_high_type": None,
        "ms_last_low_type": None,
        "ms_structure_state": "NEUTRAL",
        "ms_last_lower_high_level": np.nan,
        "ms_last_higher_low_level": np.nan,
        "msb_bull": False,
        "msb_bear": False,
        "msb_bull_active": False,
        "msb_bear_active": False,
        "msb_level": np.nan,
        "msb_confirm_bar": pd.Series([None] * len(out), index=out.index, dtype="object"),
    }
    for col, default in cols_defaults.items():
        if col not in out.columns:
            out[col] = default

    if settings is None or not settings.enabled or out.empty:
        return out

    left_bars = max(1, int(settings.left_bars))
    right_bars = max(1, int(settings.right_bars))

    high_pivot_idx, high_confirm_idx, high_prices = _confirm_swings(out["high"], left_bars, right_bars, mode="max")
    low_pivot_idx, low_confirm_idx, low_prices = _confirm_swings(out["low"], left_bars, right_bars, mode="min")

    if high_confirm_idx.size > 0:
        out.iloc[high_confirm_idx, out.columns.get_loc("swing_high_price")] = high_prices
        out.iloc[high_confirm_idx, out.columns.get_loc("swing_high_idx")] = high_pivot_idx
        out.iloc[high_confirm_idx, out.columns.get_loc("swing_high_confirmed_at")] = list(open_time.iloc[high_confirm_idx])

    if low_confirm_idx.size > 0:
        out.iloc[low_confirm_idx, out.columns.get_loc("swing_low_price")] = low_prices
        out.iloc[low_confirm_idx, out.columns.get_loc("swing_low_idx")] = low_pivot_idx
        out.iloc[low_confirm_idx, out.columns.get_loc("swing_low_confirmed_at")] = list(open_time.iloc[low_confirm_idx])

    high_type_event = pd.Series(index=out.index, dtype="object")
    low_type_event = pd.Series(index=out.index, dtype="object")
    lh_event_level = pd.Series(np.nan, index=out.index, dtype=float)
    hl_event_level = pd.Series(np.nan, index=out.index, dtype=float)

    last_high_price = np.nan
    for confirm_i, price in zip(high_confirm_idx, high_prices):
        if np.isnan(last_high_price):
            kind = None
        elif price > last_high_price:
            kind = "HH"
        else:
            kind = "LH"
            lh_event_level.iloc[confirm_i] = float(price)
        high_type_event.iloc[confirm_i] = kind
        last_high_price = float(price)

    last_low_price = np.nan
    for confirm_i, price in zip(low_confirm_idx, low_prices):
        if np.isnan(last_low_price):
            kind = None
        elif price > last_low_price:
            kind = "HL"
            hl_event_level.iloc[confirm_i] = float(price)
        else:
            kind = "LL"
        low_type_event.iloc[confirm_i] = kind
        last_low_price = float(price)

    out["ms_last_swing_high"] = out["swing_high_price"].ffill()
    out["ms_last_swing_low"] = out["swing_low_price"].ffill()
    out["ms_last_high_type"] = high_type_event.ffill()
    out["ms_last_low_type"] = low_type_event.ffill()

    bull_structure = (out["ms_last_high_type"] == "HH") & (out["ms_last_low_type"] == "HL")
    bear_structure = (out["ms_last_high_type"] == "LH") & (out["ms_last_low_type"] == "LL")
    out["ms_structure_state"] = np.where(bull_structure, "BULLISH", np.where(bear_structure, "BEARISH", "NEUTRAL"))

    out["ms_last_lower_high_level"] = lh_event_level.ffill()
    out["ms_last_higher_low_level"] = hl_event_level.ffill()

    msb_enabled = bool(settings.msb.enabled)
    if msb_enabled:
        atr = out.get("atr_14", out.get("atr14", pd.Series(np.nan, index=out.index)))
        atr = pd.to_numeric(atr, errors="coerce")
        min_break_atr = max(0.0, float(settings.msb.min_break_atr))
        break_buffer = atr.fillna(0.0) * min_break_atr

        bull_level = out["ms_last_lower_high_level"]
        bear_level = out["ms_last_higher_low_level"]

        bull_condition = ((out["close"] > (bull_level + break_buffer)) & bull_level.notna()).fillna(False).astype(bool)
        bear_condition = ((out["close"] < (bear_level - break_buffer)) & bear_level.notna()).fillna(False).astype(bool)

        bull_level_changed = bull_level.ne(bull_level.shift(1)) & bull_level.notna()
        bear_level_changed = bear_level.ne(bear_level.shift(1)) & bear_level.notna()

        prev_bull_condition = bull_condition.shift(1, fill_value=False).astype(bool)
        prev_bear_condition = bear_condition.shift(1, fill_value=False).astype(bool)

        msb_bull_event = bull_condition & (~prev_bull_condition | bull_level_changed)
        msb_bear_event = bear_condition & (~prev_bear_condition | bear_level_changed)

        out["msb_bull"] = msb_bull_event
        out["msb_bear"] = msb_bear_event

        persist = max(1, int(settings.msb.persist_bars))
        if persist > 1:
            out["msb_bull_active"] = msb_bull_event.rolling(window=persist, min_periods=1).max().astype(bool)
            out["msb_bear_active"] = msb_bear_event.rolling(window=persist, min_periods=1).max().astype(bool)
        else:
            out["msb_bull_active"] = msb_bull_event
            out["msb_bear_active"] = msb_bear_event

        out["msb_level"] = np.where(msb_bull_event, bull_level, np.where(msb_bear_event, bear_level, np.nan))
        out["msb_confirm_bar"] = pd.Series([None] * len(out), index=out.index, dtype="object")
        event_mask = msb_bull_event | msb_bear_event
        out.loc[event_mask, "msb_confirm_bar"] = list(open_time.loc[event_mask])
    else:
        out["msb_bull"] = False
        out["msb_bear"] = False
        out["msb_bull_active"] = False
        out["msb_bear_active"] = False
        out["msb_level"] = np.nan
        out["msb_confirm_bar"] = pd.Series([None] * len(out), index=out.index, dtype="object")

    return out


def build_market_structure_stats(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "count_hh": 0,
            "count_hl": 0,
            "count_lh": 0,
            "count_ll": 0,
            "msb_bull_count": 0,
            "msb_bear_count": 0,
            "structure_time_pct": {"BULLISH": 0.0, "BEARISH": 0.0, "NEUTRAL": 0.0},
            "avg_msb_level_bull": 0.0,
            "avg_msb_level_bear": 0.0,
        }

    high_event_mask = df.get("swing_high_price", pd.Series(np.nan, index=df.index)).notna()
    low_event_mask = df.get("swing_low_price", pd.Series(np.nan, index=df.index)).notna()
    high_types = df.get("ms_last_high_type", pd.Series(dtype="object")).where(high_event_mask)
    low_types = df.get("ms_last_low_type", pd.Series(dtype="object")).where(low_event_mask)
    states = df.get("ms_structure_state", pd.Series(dtype="object")).astype(str)

    state_counts = states.value_counts(normalize=True).to_dict()
    bull_msb_rows = df.get("msb_bull", pd.Series(False, index=df.index)).astype(bool)
    bear_msb_rows = df.get("msb_bear", pd.Series(False, index=df.index)).astype(bool)

    msb_level = pd.to_numeric(df.get("msb_level", pd.Series(np.nan, index=df.index)), errors="coerce")
    avg_bull = float(msb_level.loc[bull_msb_rows].mean()) if bull_msb_rows.any() else 0.0
    avg_bear = float(msb_level.loc[bear_msb_rows].mean()) if bear_msb_rows.any() else 0.0

    return {
        "count_hh": int((high_types == "HH").sum()),
        "count_hl": int((low_types == "HL").sum()),
        "count_lh": int((high_types == "LH").sum()),
        "count_ll": int((low_types == "LL").sum()),
        "msb_bull_count": int(bull_msb_rows.sum()),
        "msb_bear_count": int(bear_msb_rows.sum()),
        "structure_time_pct": {
            "BULLISH": float(state_counts.get("BULLISH", 0.0)),
            "BEARISH": float(state_counts.get("BEARISH", 0.0)),
            "NEUTRAL": float(state_counts.get("NEUTRAL", 0.0)),
        },
        "avg_msb_level_bull": avg_bull,
        "avg_msb_level_bear": avg_bear,
    }


def build_pivots_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "pivot_type", "price", "confirmed_at", "pivot_idx", "confirmed_idx"])

    rows: list[dict[str, object]] = []

    hi = df[df["swing_high_price"].notna()]
    for idx, row in hi.iterrows():
        pivot_idx = int(row.get("swing_high_idx", np.nan)) if not pd.isna(row.get("swing_high_idx", np.nan)) else int(idx)
        rows.append(
            {
                "timestamp": df.iloc[pivot_idx]["open_time"],
                "pivot_type": "HIGH",
                "price": float(row["swing_high_price"]),
                "confirmed_at": row.get("swing_high_confirmed_at"),
                "pivot_idx": pivot_idx,
                "confirmed_idx": int(idx),
            }
        )

    lo = df[df["swing_low_price"].notna()]
    for idx, row in lo.iterrows():
        pivot_idx = int(row.get("swing_low_idx", np.nan)) if not pd.isna(row.get("swing_low_idx", np.nan)) else int(idx)
        rows.append(
            {
                "timestamp": df.iloc[pivot_idx]["open_time"],
                "pivot_type": "LOW",
                "price": float(row["swing_low_price"]),
                "confirmed_at": row.get("swing_low_confirmed_at"),
                "pivot_idx": pivot_idx,
                "confirmed_idx": int(idx),
            }
        )

    pivots = pd.DataFrame(rows)
    if pivots.empty:
        return pd.DataFrame(columns=["timestamp", "pivot_type", "price", "confirmed_at", "pivot_idx", "confirmed_idx"])

    return pivots.sort_values(["confirmed_idx", "pivot_type"]).reset_index(drop=True)
