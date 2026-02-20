from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return 0.0 if abs(b) <= 1e-12 else float(a / b)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe_from_hourly_returns(ret: pd.Series) -> float:
    r = ret.dropna()
    if r.empty or float(r.std(ddof=0)) <= 1e-12:
        return 0.0
    periods = 24 * 365
    return float((r.mean() / r.std(ddof=0)) * np.sqrt(periods))


def sortino_from_hourly_returns(ret: pd.Series) -> float:
    r = ret.dropna()
    if r.empty:
        return 0.0
    downside = r[r < 0]
    if downside.empty or float(downside.std(ddof=0)) <= 1e-12:
        return 0.0
    periods = 24 * 365
    return float((r.mean() / downside.std(ddof=0)) * np.sqrt(periods))


def compute_metrics(trades: pd.DataFrame, equity_df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> dict:
    if equity_df.empty:
        return {}

    eq = equity_df["equity"].astype(float)
    ret = eq.pct_change().fillna(0.0)

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if eq.iloc[0] != 0 else 0.0
    duration_years = max(1e-9, (end_ts - start_ts).total_seconds() / (365.25 * 24 * 3600))
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / duration_years) - 1.0) if eq.iloc[0] > 0 else 0.0
    mdd = max_drawdown(eq)

    pnl = trades["pnl"].astype(float) if not trades.empty else pd.Series(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if not losses.empty else 0.0
    win_rate = float((pnl > 0).mean()) if not pnl.empty else 0.0
    expectancy = float(pnl.mean()) if not pnl.empty else 0.0
    avg_r = float(trades["R"].mean()) if "R" in trades.columns and not trades.empty else 0.0

    exposure = 0.0
    if not trades.empty and "duration_candles" in trades.columns:
        total_candles = max(1, len(equity_df))
        exposure = float(trades["duration_candles"].sum() / total_candles)

    calmar = _safe_div(cagr, abs(mdd)) if mdd < 0 else 0.0

    out = {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": mdd,
        "sharpe": sharpe_from_hourly_returns(ret),
        "sortino": sortino_from_hourly_returns(ret),
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_R": avg_r,
        "trades": int(len(trades)),
        "avg_duration_candles": float(trades["duration_candles"].mean()) if not trades.empty else 0.0,
        "exposure": exposure,
        "long_trades": int((trades.get("side", pd.Series(dtype=str)) == "LONG").sum()) if not trades.empty else 0,
        "short_trades": int((trades.get("side", pd.Series(dtype=str)) == "SHORT").sum()) if not trades.empty else 0,
    }

    if not trades.empty and "entry_time" in trades.columns:
        trades_year = trades.copy()
        trades_year["entry_time"] = pd.to_datetime(trades_year["entry_time"], utc=True)
        year_rows = {}
        for year, g in trades_year.groupby(trades_year["entry_time"].dt.year):
            year_rows[str(int(year))] = {
                "trades": int(len(g)),
                "pnl": float(g["pnl"].sum()),
                "win_rate": float((g["pnl"] > 0).mean()),
                "profit_factor": float(g.loc[g["pnl"] > 0, "pnl"].sum() / abs(g.loc[g["pnl"] < 0, "pnl"].sum()))
                if (g["pnl"] < 0).any()
                else 0.0,
            }
        out["by_year"] = year_rows
    else:
        out["by_year"] = {}

    return out
