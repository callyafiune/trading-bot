from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

BINANCE_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
ALTERNATIVE_ME_BASE = "https://api.alternative.me"

INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
}


class BinanceDataClient:
    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    @staticmethod
    def _to_ms(value: str) -> int:
        return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)

    def fetch_fear_greed(self, start_date: str, end_date: str) -> pd.DataFrame:
        resp = self.session.get(f"{ALTERNATIVE_ME_BASE}/fng/", params={"limit": 0, "format": "json"}, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data", [])
        if not rows:
            return pd.DataFrame(columns=["open_time", "fng_value", "classification"])

        df = pd.DataFrame(rows)
        df["open_time"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.floor("D")
        df["fng_value"] = pd.to_numeric(df["value"], errors="coerce")
        df["classification"] = df.get("value_classification", "").astype(str)
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")
        df = df[(df["open_time"] >= start_ts) & (df["open_time"] <= end_ts)]
        df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
        return df[["open_time", "fng_value", "classification"]].reset_index(drop=True)


    def fetch_funding_rate(self, symbol: str, start_date: str, end_date: str, limit: int = 1000) -> pd.DataFrame:
        start_ms = self._to_ms(start_date)
        end_ms = self._to_ms(end_date)
        rows: list[dict[str, Any]] = []

        while start_ms < end_ms:
            params = {
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
            resp = self.session.get(f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate", params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            rows.extend(batch)
            last_ts = int(batch[-1]["fundingTime"])
            start_ms = last_ts + 1
            time.sleep(0.1)

        if not rows:
            return pd.DataFrame(columns=["funding_time", "funding_rate", "mark_price"])

        df = pd.DataFrame(rows)
        df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df["mark_price"] = pd.to_numeric(df.get("markPrice"), errors="coerce")
        df = df[["funding_time", "funding_rate", "mark_price"]].sort_values("funding_time")
        return df.reset_index(drop=True)
    def fetch_ohlcv(self, symbol: str, interval: str, start_date: str, end_date: str, limit: int = 1000) -> pd.DataFrame:
        start_ms = self._to_ms(start_date)
        end_ms = self._to_ms(end_date)
        step = INTERVAL_MS[interval]
        rows: list[list[Any]] = []

        while start_ms < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
            resp = self.session.get(f"{BINANCE_BASE}/api/v3/klines", params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            rows.extend(batch)
            start_ms = int(batch[-1][0]) + step
            time.sleep(0.1)

        cols = [
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore",
        ]
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return df
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df[["open_time", "open", "high", "low", "close", "volume"]].sort_values("open_time")
        return df.reset_index(drop=True)
