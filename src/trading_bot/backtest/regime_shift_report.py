from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_regime_shift_outputs(
    out_dir: str | Path,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    report: dict,
    include_plots: bool = True,
) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    trades.to_csv(p / "trades.csv", index=False)
    eq = equity.copy()
    if not eq.empty and "equity" in eq.columns:
        eq["drawdown"] = eq["equity"].astype(float) / eq["equity"].astype(float).cummax() - 1.0
    eq.to_csv(p / "equity.csv", index=False)
    (p / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if include_plots:
        _plot_equity(equity, p / "equity.png")
        _plot_drawdown(equity, p / "drawdown.png")
        _plot_hist_r(trades, p / "hist_R.png")
        _plot_trades_per_year(trades, p / "trades_per_year.png")


def _plot_equity(equity: pd.DataFrame, path: Path) -> None:
    if equity.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(equity["timestamp"], utc=True), equity["equity"], lw=1.3)
    ax.set_title("Equity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_drawdown(equity: pd.DataFrame, path: Path) -> None:
    if equity.empty:
        return
    eq = equity["equity"].astype(float)
    dd = eq / eq.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(pd.to_datetime(equity["timestamp"], utc=True), dd, 0.0, alpha=0.35)
    ax.set_title("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_hist_r(trades: pd.DataFrame, path: Path) -> None:
    if trades.empty or "R" not in trades.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(trades["R"].astype(float), bins=30, alpha=0.75)
    ax.set_title("R Histogram")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_trades_per_year(trades: pd.DataFrame, path: Path) -> None:
    if trades.empty or "entry_time" not in trades.columns:
        return
    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    by_year = t.groupby(t["entry_time"].dt.year).size()
    fig, ax = plt.subplots(figsize=(8, 4))
    by_year.plot(kind="bar", ax=ax)
    ax.set_title("Trades per Year")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
