from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.backtest.metrics import compute_metrics
from trading_bot.backtest.regime_shift_engine import run_regime_shift_backtest
from trading_bot.backtest.regime_shift_report import save_regime_shift_outputs
from trading_bot.data.regime_shift_loader import load_ohlcv
from trading_bot.regime_shift_config import RegimeShiftConfig, build_grid_configs, load_regime_shift_config
from trading_bot.strategies.regime_shift_major_levels import detect_regime_shift_levels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest regime-shift major levels")
    p.add_argument("--config", required=True)
    p.add_argument("--force-refresh", action="store_true")
    return p.parse_args()


def scenario_name(cfg: RegimeShiftConfig) -> str:
    sig = f"atr{cfg.atr_k}" if cfg.significance_mode == "atr" else f"pct{cfg.percent_p}"
    buf = f"atr{cfg.atr_buffer_k}" if cfg.buffer_mode == "atr" else f"pct{cfg.percent_buffer_p}"
    tp = f"rr{cfg.rr_target}" if cfg.tp_mode == "rr" else "trail"
    stop = f"atr{cfg.stop_atr_k}" if cfg.stop_mode == "atr" else "major"
    return (
        f"lb{cfg.lookback_init}_{cfg.significance_mode}_{sig}_{cfg.break_confirm_mode}_{cfg.buffer_mode}_{buf}_"
        f"stop{stop}_tp{tp}_flip{int(cfg.allow_flip)}_ma99{int(cfg.filter_ma99)}"
    )


def top_drawdown_periods(equity: pd.DataFrame, top_n: int = 3) -> list[dict]:
    if equity.empty:
        return []
    e = equity.copy()
    e["equity"] = e["equity"].astype(float)
    e["dd"] = e["equity"] / e["equity"].cummax() - 1.0
    worst = e.nsmallest(top_n, "dd")
    return [{"timestamp": str(r["timestamp"]), "drawdown": float(r["dd"])} for _, r in worst.iterrows()]


def run_one(df: pd.DataFrame, cfg: RegimeShiftConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    feat = detect_regime_shift_levels(df, cfg)
    trades, equity = run_regime_shift_backtest(feat, cfg)

    start = pd.Timestamp(feat["open_time"].min())
    end = pd.Timestamp(feat["open_time"].max())
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    metrics = compute_metrics(trades, equity, start, end)
    return trades, equity, metrics


def main() -> None:
    args = parse_args()
    cfg = load_regime_shift_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    np.random.seed(int(cfg.seed))

    t0 = time.time()
    df = load_ohlcv(cfg, force_refresh=bool(args.force_refresh))
    if df.empty:
        raise RuntimeError("No OHLCV data")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_root = Path(cfg.output_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    grid = build_grid_configs(cfg)
    rows: list[dict] = []
    best = None

    for i, g in enumerate(grid, start=1):
        tr, eq, met = run_one(df, g)
        scen = scenario_name(g)
        scen_dir = out_root / "scenarios" / scen
        save_regime_shift_outputs(scen_dir, tr, eq, met, include_plots=False)

        row = {
            "scenario": scen,
            "lookback_init": g.lookback_init,
            "significance_mode": g.significance_mode,
            "atr_k": g.atr_k,
            "percent_p": g.percent_p,
            "break_confirm_mode": g.break_confirm_mode,
            "buffer_mode": g.buffer_mode,
            "atr_buffer_k": g.atr_buffer_k,
            "percent_buffer_p": g.percent_buffer_p,
            "stop_mode": g.stop_mode,
            "stop_atr_k": g.stop_atr_k,
            "tp_mode": g.tp_mode,
            "rr_target": g.rr_target,
            "allow_flip": g.allow_flip,
            "filter_ma99": g.filter_ma99,
            "calmar": float(met.get("calmar", 0.0)),
            "profit_factor": float(met.get("profit_factor", 0.0)),
            "expectancy": float(met.get("expectancy", 0.0)),
            "max_drawdown": float(met.get("max_drawdown", 0.0)),
            "trades": int(met.get("trades", 0)),
        }
        rows.append(row)

        if best is None:
            best = (g, tr, eq, met, row)
        else:
            _, _, _, _, b = best
            if (row["calmar"], row["profit_factor"]) > (b["calmar"], b["profit_factor"]):
                best = (g, tr, eq, met, row)

        if i % 20 == 0:
            logging.info("grid_progress=%s/%s", i, len(grid))

    grid_df = pd.DataFrame(rows).sort_values(["calmar", "profit_factor"], ascending=[False, False]).reset_index(drop=True)
    grid_df.to_csv(out_root / "grid_results.csv", index=False)

    assert best is not None
    best_cfg, best_tr, best_eq, best_met, best_row = best
    save_regime_shift_outputs(out_root / "best", best_tr, best_eq, best_met, include_plots=True)

    worst_dd = top_drawdown_periods(best_eq, top_n=3)
    summary = {
        "run_id": run_id,
        "best_scenario": best_row,
        "best_metrics": best_met,
        "worst_drawdown_periods": worst_dd,
        "runtime_seconds": time.time() - t0,
        "grid_size": len(grid),
    }
    (out_root / "report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    top5 = grid_df.head(5)
    print("run_id=", run_id)
    print("out_dir=", out_root)
    print("Top 5 by Calmar/ProfitFactor")
    print(top5.to_string(index=False))
    print("worst_drawdown_periods=", worst_dd)


if __name__ == "__main__":
    main()
