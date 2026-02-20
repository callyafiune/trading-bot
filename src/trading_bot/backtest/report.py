from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_outputs(
    out_dir: str | Path,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    metrics: dict,
    grid_results: pd.DataFrame,
) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    trades.to_csv(target / "trades.csv", index=False)
    equity.to_csv(target / "equity_curve.csv", index=False)
    grid_results.to_csv(target / "grid_results.csv", index=False)
    (target / "summary.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    _plot_equity(equity, target / "equity_curve.png")
    _plot_drawdown(equity, target / "drawdown.png")
    _plot_r_hist(trades, target / "r_histogram.png")
    _plot_trades_by_year(trades, target / "trades_by_year.png")


def _plot_equity(equity: pd.DataFrame, path: Path) -> None:
    if equity.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pd.to_datetime(equity["timestamp"], utc=True), equity["equity"], label="Equity")
    ax.set_title("Equity Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_drawdown(equity: pd.DataFrame, path: Path) -> None:
    if equity.empty:
        return
    eq = equity["equity"].astype(float)
    dd = eq / eq.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(pd.to_datetime(equity["timestamp"], utc=True), dd, 0.0, alpha=0.35)
    ax.set_title("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_r_hist(trades: pd.DataFrame, path: Path) -> None:
    if trades.empty or "R" not in trades.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(trades["R"].astype(float), bins=30, alpha=0.75)
    ax.set_title("R Distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_trades_by_year(trades: pd.DataFrame, path: Path) -> None:
    if trades.empty or "entry_time" not in trades.columns:
        return
    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    by_year = t.groupby(t["entry_time"].dt.year).size()
    fig, ax = plt.subplots(figsize=(8, 4))
    by_year.plot(kind="bar", ax=ax)
    ax.set_title("Trades by Year")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
