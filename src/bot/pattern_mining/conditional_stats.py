from __future__ import annotations

import pandas as pd


def build_conditional_stats(
    df: pd.DataFrame,
    horizon: int = 3,
    events: list[str] | None = None,
) -> pd.DataFrame:
    event_cols = events or [
        "big_lower_wick",
        "sweep_down_reclaim",
        "break_structure_up",
        "break_structure_down",
        "regime_above_ma99",
        "regime_ma99_slope_up",
    ]

    ret_col = f"future_return_{horizon}h"
    rows: list[dict[str, float | int | str]] = []
    for event_col in event_cols:
        if event_col not in df.columns:
            continue
        subset = df[df[event_col].fillna(False)]
        count = int(len(subset))
        if count == 0:
            rows.append(
                {
                    "event": event_col,
                    "count": 0,
                    "prob_y_up": 0.0,
                    "prob_y_down": 0.0,
                    f"avg_future_return_{horizon}h": 0.0,
                }
            )
            continue

        rows.append(
            {
                "event": event_col,
                "count": count,
                "prob_y_up": float(subset["y_up"].mean()),
                "prob_y_down": float(subset["y_down"].mean()),
                f"avg_future_return_{horizon}h": float(subset[ret_col].mean()),
            }
        )

    return pd.DataFrame(rows)
