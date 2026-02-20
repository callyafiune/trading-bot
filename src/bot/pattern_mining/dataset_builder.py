from __future__ import annotations

from typing import Iterable

import pandas as pd


TIMESTAMP_CANDIDATES = ("open_time", "timestamp", "ts", "datetime", "date")


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts_col = next((c for c in TIMESTAMP_CANDIDATES if c in out.columns), None)
    if ts_col is None:
        raise ValueError("DataFrame sem coluna de timestamp reconhecida")
    if ts_col != "open_time":
        out = out.rename(columns={ts_col: "open_time"})
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    return out


def _normalize_optional(
    df: pd.DataFrame | None,
    value_col_candidates: Iterable[str],
    out_col: str,
) -> pd.DataFrame | None:
    if df is None:
        return None
    if df.empty:
        return pd.DataFrame(columns=["open_time", out_col])
    out = _normalize_timestamp(df)
    src_col = next((c for c in value_col_candidates if c in out.columns), None)
    if src_col is None:
        return pd.DataFrame(columns=["open_time", out_col])
    out = out[["open_time", src_col]].rename(columns={src_col: out_col})
    return out


def build_synchronized_dataset(
    ohlcv: pd.DataFrame,
    oi: pd.DataFrame | None = None,
    cvd: pd.DataFrame | None = None,
    liquidations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Builds a timestamp-aligned dataset without introducing future leakage.

    The OHLCV timeline is treated as the reference index; optional series are merged
    by exact timestamp and only forward-filled (never backfilled).
    """

    if ohlcv.empty:
        return ohlcv.copy()

    base = _normalize_timestamp(ohlcv)
    merged = base.copy()

    optional_blocks = [
        _normalize_optional(oi, ("oi", "open_interest", "value"), "oi"),
        _normalize_optional(cvd, ("cvd", "delta", "value"), "cvd"),
        _normalize_optional(liquidations, ("liquidations", "liq", "value"), "liquidations"),
    ]

    for block in optional_blocks:
        if block is None:
            continue
        merged = merged.merge(block, on="open_time", how="left")

    for col in ("oi", "cvd", "liquidations"):
        if col in merged.columns:
            merged[col] = merged[col].ffill()

    return merged
