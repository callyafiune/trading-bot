from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtests.baseline import prepare_pattern_frame
from backtests.config import ValidationConfig
from backtests.event_study import compute_event_statistics
from backtests.metrics import write_json
from bot.pattern_mining.feature_engineering import build_regime_flags


def _build_regimes(frame: pd.DataFrame, cfg: ValidationConfig) -> pd.DataFrame:
    _ = cfg
    return build_regime_flags(frame)


def run_regime_event_study(df: pd.DataFrame, cfg: ValidationConfig, prepared_df: pd.DataFrame | None = None) -> dict:
    frame = prepared_df.copy() if prepared_df is not None else prepare_pattern_frame(df, cfg)
    frame = _build_regimes(frame, cfg)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regime_summary = frame["regime_id"].value_counts().astype(int).to_dict()
    write_json(out_dir / "regime_summary.json", {"counts": regime_summary})

    all_rows = []
    for regime_id, group in frame.groupby("regime_id"):
        stats = compute_event_statistics(group, cfg, min_support=max(10, cfg.min_support // 4))
        if stats.empty:
            continue
        stats = stats.copy()
        stats["regime_id"] = regime_id
        stats["regime_price_above_ma99"] = int(group["regime_price_above_ma99"].iloc[0])
        stats["regime_ma99_slope_up"] = int(group["regime_ma99_slope_up"].iloc[0])
        stats["regime_vol_high"] = int(group["regime_vol_high"].iloc[0])
        all_rows.append(stats)

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    combined.to_csv(out_dir / "regime_event_study.csv", index=False)

    best_by_regime: dict[str, dict] = {}
    if not combined.empty:
        for regime_id, rg in combined.groupby("regime_id"):
            best_by_regime[str(regime_id)] = rg.sort_values("edge", ascending=False).iloc[0].to_dict()

    return {
        "regime_summary": regime_summary,
        "regime_event_study": combined,
        "best_by_regime": best_by_regime,
    }
