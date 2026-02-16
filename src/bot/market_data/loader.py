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


def save_parquet(df: pd.DataFrame, raw_path: Path, processed_path: Path, interval: str) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path, index=False)
    enforce_continuous_candles(df, interval).to_parquet(processed_path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
