from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> dict:
    if equity_curve.empty:
        return {
            "return_net": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "turnover": 0.0,
            "avg_hours_in_pos": 0.0,
        }

    eq = equity_curve["equity"] if "equity" in equity_curve.columns else pd.Series(dtype=float)
    if eq.empty:
        return {
            "return_net": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "turnover": 0.0,
            "avg_hours_in_pos": 0.0,
        }

    if trades.empty:
        dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
        return {
            "return_net": 0.0,
            "max_drawdown": float(dd),
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "turnover": 0.0,
            "avg_hours_in_pos": 0.0,
        }

    total_pnl = trades["pnl_net"].sum()
    gross_profit = trades.loc[trades["pnl_net"] > 0, "pnl_net"].sum()
    gross_loss = trades.loc[trades["pnl_net"] < 0, "pnl_net"].sum()
    win_rate = (trades["pnl_net"] > 0).mean()
    expectancy = trades["pnl_net"].mean()
    turnover = trades["notional"].sum() / max(eq.iloc[0], 1e-9)

    dd = (eq / eq.cummax() - 1).min()
    returns = eq.pct_change().dropna()
    sharpe = np.sqrt(24 * 365) * returns.mean() / (returns.std() + 1e-9) if not returns.empty else 0.0
    return {
        "return_net": float(total_pnl / max(eq.iloc[0], 1e-9)),
        "max_drawdown": float(dd),
        "profit_factor": float(gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf),
        "sharpe": float(sharpe),
        "win_rate": float(win_rate),
        "expectancy": float(expectancy),
        "turnover": float(turnover),
        "avg_hours_in_pos": float(trades["hold_hours"].mean()),
    }
