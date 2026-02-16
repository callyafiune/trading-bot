from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

BINANCE_BASE = "https://api.binance.com"

INTERVAL_MS = {
    "1h": 60 * 60 * 1000,
}


class BinanceDataClient:
    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    @staticmethod
    def _to_ms(value: str) -> int:
        return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)

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
