from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtests.config import ValidationConfig
from backtests.metrics import write_json


def _timeframe_delta_seconds(timeframe: str) -> float | None:
    try:
        td = pd.to_timedelta(timeframe)
        return float(td.total_seconds())
    except Exception:
        return None


def run_sanity_checks(df: pd.DataFrame, cfg: ValidationConfig) -> dict:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = df.copy().sort_values("open_time").reset_index(drop=True)
    duplicates = int(frame.duplicated(subset=["open_time"]).sum())

    deltas = frame["open_time"].diff().dropna().dt.total_seconds()
    expected_delta = _timeframe_delta_seconds(cfg.timeframe)
    gap_count = None
    if expected_delta is not None and not deltas.empty:
        gap_count = int((deltas != expected_delta).sum())

    nan_by_column = {k: int(v) for k, v in frame.isna().sum().to_dict().items()}

    checks = {
        "high_lt_low": int((frame["high"] < frame["low"]).sum()),
        "volume_lt_0": int((frame["volume"] < 0).sum()),
        "close_outside_low_high": int(((frame["close"] < frame["low"]) | (frame["close"] > frame["high"])).sum()),
    }

    range_series = frame["high"] - frame["low"]
    return_1 = np.log(frame["close"]).diff()
    stat_columns: dict[str, pd.Series] = {
        "range": range_series,
        "return_1": return_1,
        "volume": frame["volume"],
    }

    if "oi" in frame.columns:
        stat_columns["var_oi_1"] = frame["oi"].pct_change(1)
    if "cvd" in frame.columns:
        stat_columns["var_cvd_1"] = frame["cvd"].pct_change(1)

    quantiles = [0.01, 0.05, 0.50, 0.95, 0.99]
    stats: dict[str, dict[str, float]] = {}
    for name, series in stat_columns.items():
        s = pd.Series(series, dtype=float).dropna()
        if s.empty:
            stats[name] = {"min": np.nan, "max": np.nan, "q01": np.nan, "q05": np.nan, "q50": np.nan, "q95": np.nan, "q99": np.nan}
            continue
        q = s.quantile(quantiles)
        stats[name] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "q01": float(q.loc[0.01]),
            "q05": float(q.loc[0.05]),
            "q50": float(q.loc[0.50]),
            "q95": float(q.loc[0.95]),
            "q99": float(q.loc[0.99]),
        }

    delta_percentiles = {}
    if not deltas.empty:
        dq = deltas.quantile([0.01, 0.05, 0.50, 0.95, 0.99])
        delta_percentiles = {
            "q01": float(dq.loc[0.01]),
            "q05": float(dq.loc[0.05]),
            "q50": float(dq.loc[0.50]),
            "q95": float(dq.loc[0.95]),
            "q99": float(dq.loc[0.99]),
        }

    report = {
        "n_rows": int(len(frame)),
        "ts_min": str(frame["open_time"].min()) if not frame.empty else None,
        "ts_max": str(frame["open_time"].max()) if not frame.empty else None,
        "duplicates": duplicates,
        "gaps": {
            "expected_delta_seconds": expected_delta,
            "gap_count": gap_count,
            "delta_percentiles_seconds": delta_percentiles,
        },
        "nan_by_column": nan_by_column,
        "checks": checks,
        "stats": stats,
    }

    write_json(out_dir / "sanity_report.json", report)
    frame.head(50).to_csv(out_dir / "sanity_preview_head.csv", index=False)
    frame.tail(50).to_csv(out_dir / "sanity_preview_tail.csv", index=False)
    return report
