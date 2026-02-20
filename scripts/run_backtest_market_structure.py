from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.backtest.engine import run_backtest
from trading_bot.backtest.metrics import compute_metrics
from trading_bot.backtest.report import save_outputs
from trading_bot.config import MarketStructureConfig, build_grid_configs, load_config
from trading_bot.data.loader import load_ohlcv
from trading_bot.strategies.market_structure_hhhl import generate_structure_signals, prepare_structure_features


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run market-structure HH/HL backtest")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--force-refresh", action="store_true", help="Force download instead of cache")
    return p.parse_args()


def _scenario_name(cfg: MarketStructureConfig) -> str:
    return (
        f"pl{cfg.pivot_left}_pr{cfg.pivot_right}_tp{cfg.tp_mode}_rr{cfg.rr_target}_"
        f"stop{cfg.stop_mode}_atr{cfg.atr_mult}_ma99{int(cfg.filter_ma99)}"
    )


def _drawdown_periods(equity_df: pd.DataFrame, top_n: int = 3) -> list[dict]:
    if equity_df.empty:
        return []
    eq = equity_df.copy()
    eq["equity"] = eq["equity"].astype(float)
    eq["peak"] = eq["equity"].cummax()
    eq["drawdown"] = eq["equity"] / eq["peak"] - 1.0
    worst = eq.nsmallest(top_n, "drawdown")
    rows = []
    for _, r in worst.iterrows():
        rows.append({"timestamp": str(r["timestamp"]), "drawdown": float(r["drawdown"])})
    return rows


def _run_single(df: pd.DataFrame, cfg: MarketStructureConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    feat_df, _ = prepare_structure_features(df, cfg)
    signals = generate_structure_signals(feat_df, cfg)
    trades, equity = run_backtest(feat_df, signals, cfg)
    start = pd.Timestamp(feat_df["open_time"].min()) if not feat_df.empty else pd.Timestamp.now(tz="UTC")
    end = pd.Timestamp(feat_df["open_time"].max()) if not feat_df.empty else pd.Timestamp.now(tz="UTC")
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    metrics = compute_metrics(
        trades,
        equity,
        start_ts=start,
        end_ts=end,
    )
    return trades, equity, metrics


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    np.random.seed(int(cfg.seed))

    df = load_ohlcv(cfg, force_refresh=bool(args.force_refresh))
    if df.empty:
        raise RuntimeError("No OHLCV data loaded.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_root = Path(cfg.output_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    grid_cfgs = build_grid_configs(cfg)
    grid_rows: list[dict] = []
    best = None

    for i, gcfg in enumerate(grid_cfgs, start=1):
        trades, equity, metrics = _run_single(df, gcfg)
        scenario = _scenario_name(gcfg)
        scen_dir = out_root / "scenarios" / scenario
        scen_dir.mkdir(parents=True, exist_ok=True)
        trades.to_csv(scen_dir / "trades.csv", index=False)
        equity.to_csv(scen_dir / "equity_curve.csv", index=False)
        (scen_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

        row = {
            "scenario": scenario,
            "pivot_left": gcfg.pivot_left,
            "pivot_right": gcfg.pivot_right,
            "tp_mode": gcfg.tp_mode,
            "rr_target": gcfg.rr_target,
            "stop_mode": gcfg.stop_mode,
            "atr_mult": gcfg.atr_mult,
            "filter_ma99": gcfg.filter_ma99,
            "calmar": float(metrics.get("calmar", 0.0)),
            "profit_factor": float(metrics.get("profit_factor", 0.0)),
            "expectancy": float(metrics.get("expectancy", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "trades": int(metrics.get("trades", 0)),
        }
        grid_rows.append(row)

        if best is None:
            best = (gcfg, trades, equity, metrics, row)
        else:
            _, _, _, _, best_row = best
            cur_key = (row["calmar"], row["profit_factor"])
            best_key = (best_row["calmar"], best_row["profit_factor"])
            if cur_key > best_key:
                best = (gcfg, trades, equity, metrics, row)

        if i % 10 == 0:
            print(f"grid_progress={i}/{len(grid_cfgs)}")

    grid_df = pd.DataFrame(grid_rows).sort_values(["calmar", "profit_factor"], ascending=[False, False])
    grid_df.to_csv(out_root / "grid_results.csv", index=False)

    assert best is not None
    best_cfg, best_trades, best_equity, best_metrics, best_row = best

    save_outputs(
        out_dir=out_root / "best",
        trades=best_trades,
        equity=best_equity.rename(columns={"timestamp": "timestamp", "equity": "equity"}),
        metrics=best_metrics,
        grid_results=grid_df,
    )

    dd_periods = _drawdown_periods(best_equity)
    summary = {
        "run_id": run_id,
        "best_scenario": best_row,
        "best_metrics": best_metrics,
        "worst_drawdown_periods": dd_periods,
        "data_start": str(df["open_time"].min()),
        "data_end": str(df["open_time"].max()),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("run_id=", run_id)
    print("out_dir=", out_root)
    print("best_scenario=", best_row["scenario"])
    print("best_calmar=", f"{best_row['calmar']:.6f}", "best_profit_factor=", f"{best_row['profit_factor']:.6f}")
    print("worst_drawdown_periods=", dd_periods)


if __name__ == "__main__":
    main()
