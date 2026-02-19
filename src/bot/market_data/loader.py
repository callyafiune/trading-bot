from __future__ import annotations

from pathlib import Path

import pandas as pd

from bot.features.builder import build_features
from bot.market_data.binance_client import INTERVAL_MS
from bot.utils.config import FeatureSettings, MarketStructureSettings


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if timeframe.lower() != "4h":
        raise ValueError(f"timeframe não suportado para resample_ohlcv: {timeframe}")

    base = df.sort_values("open_time").set_index("open_time")
    agg = (
        base.resample("4h", label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return agg


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


def save_parquet(
    df: pd.DataFrame,
    raw_path: Path,
    processed_path: Path,
    interval: str,
    feature_settings: FeatureSettings | None = None,
    market_structure_settings: MarketStructureSettings | None = None,
) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path, index=False)
    processed = enforce_continuous_candles(df, interval)
    processed = build_features(processed, feature_settings, market_structure_settings=market_structure_settings)
    processed.to_parquet(processed_path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def derive_detail_data_path(data_path: str | Path, detail_timeframe: str) -> Path | None:
    p = Path(data_path)
    token = "_1h_"
    tf_token = f"_{detail_timeframe}_"
    if token not in p.name:
        return None
    return p.with_name(p.name.replace(token, tf_token, 1))


def merge_fng_with_ohlcv(ohlcv: pd.DataFrame, fng_1d: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return ohlcv.copy()
    if fng_1d.empty:
        out = ohlcv.copy()
        out["fng_value"] = pd.NA
        return out

    fng = fng_1d.copy()
    if "ts" in fng.columns and "open_time" not in fng.columns:
        fng = fng.rename(columns={"ts": "open_time"})
    if "timestamp" in fng.columns and "open_time" not in fng.columns:
        fng = fng.rename(columns={"timestamp": "open_time"})
    fng = fng.sort_values("open_time")
    out = ohlcv.sort_values("open_time").merge(fng[["open_time", "fng_value"]], on="open_time", how="left")
    out["fng_value"] = out["fng_value"].ffill()
    return out


def process_fng_to_1h(fng_daily: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    idx = pd.date_range(start=start_ts, end=end_ts, freq="1h", inclusive="left")
    if fng_daily.empty:
        return pd.DataFrame({"ts": idx, "fng_value": pd.NA})

    base = fng_daily.copy()
    if "timestamp" in base.columns and "open_time" not in base.columns:
        base = base.rename(columns={"timestamp": "open_time"})
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True).dt.floor("D")
    base = base.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    hourly = (
        base[["open_time", "fng_value"]]
        .set_index("open_time")
        .reindex(idx, method="ffill")
        .rename_axis("ts")
        .reset_index()
    )
    return hourly
