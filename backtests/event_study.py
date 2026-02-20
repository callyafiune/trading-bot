from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.baseline import prepare_pattern_frame
from backtests.config import ValidationConfig
from bot.pattern_mining.event_detection import detect_events

EVENT_COLUMNS = [
    "big_lower_wick",
    "sweep_down_reclaim",
    "break_structure_up",
    "break_structure_down",
    "regime_above_ma99",
    "regime_ma99_slope_up",
]


def compute_event_statistics(frame: pd.DataFrame, cfg: ValidationConfig, min_support: int | None = None) -> pd.DataFrame:
    support = int(cfg.min_support if min_support is None else min_support)
    ret_col = f"future_return_{cfg.horizon_candles}h"
    rows: list[dict] = []

    event_defs: list[tuple[str, pd.Series]] = []
    for event in EVENT_COLUMNS:
        if event in frame.columns:
            event_defs.append((event, frame[event].fillna(False).astype(bool)))

    for (e1, m1), (e2, m2) in combinations(event_defs, 2):
        event_defs.append((f"{e1}&{e2}", (m1 & m2)))

    for name, mask in event_defs:
        subset = frame[mask & frame[ret_col].notna()]
        count = int(len(subset))
        if count < support:
            continue

        returns = subset[ret_col].astype(float)
        row = {
            "event": name,
            "count": count,
            "p_y_up": float(subset["y_up"].mean()),
            "p_y_down": float(subset["y_down"].mean()),
            "mean_return": float(returns.mean()),
            "median_return": float(returns.median()),
            "p05_return": float(returns.quantile(0.05)),
            "p95_return": float(returns.quantile(0.95)),
            "mean_max_drawdown": float(subset["max_drawdown_future"].mean()),
            "mean_max_runup": float(subset["max_runup_future"].mean()),
        }
        row["edge"] = row["p_y_up"] - row["p_y_down"]
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["edge", "count"], ascending=[False, False]).reset_index(drop=True)


def run_event_study(df: pd.DataFrame, cfg: ValidationConfig, prepared_df: pd.DataFrame | None = None) -> dict:
    frame = prepared_df.copy() if prepared_df is not None else prepare_pattern_frame(df, cfg)
    if "big_lower_wick" not in frame.columns:
        frame = detect_events(frame, cfg.wick_threshold, cfg.sweep_lookback, cfg.sweep_lookback)

    stats = compute_event_statistics(frame, cfg)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats.to_csv(out_dir / "event_study.csv", index=False)
    top = stats.head(20)
    lines = []
    for idx, row in top.iterrows():
        lines.append(
            f"{idx + 1:02d}. {row['event']} | edge={row['edge']:.4f} | support={int(row['count'])} | "
            f"p_up={row['p_y_up']:.3f} | p_down={row['p_y_down']:.3f} | mean_ret={row['mean_return']:.4f}"
        )
    (out_dir / "event_study_top20.txt").write_text("\n".join(lines), encoding="utf-8")

    best_event = top.iloc[0].to_dict() if not top.empty else None
    return {
        "event_study": stats,
        "best_event": best_event,
        "frame": frame,
    }
