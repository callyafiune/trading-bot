from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series | np.ndarray | list[float]) -> float:
    series = pd.Series(equity, dtype=float).dropna()
    if series.empty:
        return 0.0
    peak = series.cummax()
    dd = (series / peak) - 1.0
    return float(dd.min())


def profit_factor(pnls: pd.Series | np.ndarray | list[float]) -> float:
    values = pd.Series(pnls, dtype=float).dropna()
    if values.empty:
        return 0.0
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(values[values < 0].sum())
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / abs(gross_loss)


def expectancy(pnls: pd.Series | np.ndarray | list[float]) -> float:
    values = pd.Series(pnls, dtype=float).dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def winrate(pnls: pd.Series | np.ndarray | list[float]) -> float:
    values = pd.Series(pnls, dtype=float).dropna()
    if values.empty:
        return 0.0
    return float((values > 0).mean())


def summarize_trades(df_trades: pd.DataFrame) -> dict[str, float | int]:
    if df_trades.empty:
        return {
            "trades": 0,
            "winrate": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_return_per_trade": 0.0,
        }

    pnl_col = "pnl" if "pnl" in df_trades.columns else "return"
    pnls = df_trades[pnl_col].astype(float)
    equity = (1.0 + pnls).cumprod()
    return {
        "trades": int(len(df_trades)),
        "winrate": winrate(pnls),
        "expectancy": expectancy(pnls),
        "profit_factor": profit_factor(pnls),
        "max_drawdown": max_drawdown(equity),
        "avg_return_per_trade": float(pnls.mean()),
    }


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
