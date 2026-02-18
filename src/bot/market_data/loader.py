from __future__ import annotations

from pathlib import Path

import pandas as pd

from bot.market_data.binance_client import INTERVAL_MS


def enforce_continuous_candles(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df
    freq_ms = INTERVAL_MS[interval]
    full_idx = pd.date_range(df["open_time"].min(), df["open_time"].max(), freq=f"{freq_ms}ms", tz="UTC")
    out = df.set_index("open_time").reindex(full_idx).rename_axis("open_time").reset_index()
    return out


def process_funding_to_1h(funding_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if funding_df.empty:
        idx = pd.date_range(start=start_date, end=end_date, freq="1h", tz="UTC", inclusive="left")
        return pd.DataFrame({"open_time": idx, "funding_rate": 0.0, "funding_missing": 1})

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    idx = pd.date_range(start=start_ts, end=end_ts, freq="1h", inclusive="left")
    hourly = (
        funding_df[["funding_time", "funding_rate"]]
        .drop_duplicates(subset=["funding_time"])
        .set_index("funding_time")
        .sort_index()
        .reindex(idx)
    )
    missing_before_ffill = hourly["funding_rate"].isna()
    hourly["funding_rate"] = hourly["funding_rate"].ffill().fillna(0.0)
    hourly["funding_missing"] = missing_before_ffill.astype(int)
    return hourly.rename_axis("open_time").reset_index()


def merge_ohlcv_with_funding(ohlcv_1h: pd.DataFrame, funding_1h: pd.DataFrame) -> pd.DataFrame:
    merged = ohlcv_1h.merge(funding_1h, on="open_time", how="left")
    if "funding_rate" not in merged.columns:
        merged["funding_rate"] = 0.0
        merged["funding_missing"] = 1
        return merged

    missing_pre = merged["funding_rate"].isna()
    merged["funding_rate"] = merged["funding_rate"].ffill().fillna(0.0)
    merged["funding_missing"] = merged.get("funding_missing", 0)
    merged["funding_missing"] = merged["funding_missing"].fillna(0).astype(int)
    merged.loc[missing_pre, "funding_missing"] = 1
    return merged


def save_parquet(df: pd.DataFrame, raw_path: Path, processed_path: Path, interval: str) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path, index=False)
    enforce_continuous_candles(df, interval).to_parquet(processed_path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
