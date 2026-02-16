from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


def save_trades_csv(trades: pd.DataFrame, path: str | Path = "data/processed/trades.csv") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(p, index=False)


def print_summary(metrics: dict) -> None:
    table = Table(title="Backtest Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.6f}" if isinstance(v, float) else str(v))
    Console().print(table)
