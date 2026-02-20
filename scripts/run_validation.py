from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.baseline import generate_baseline_signals, prepare_pattern_frame, run_baseline
from backtests.config import ValidationConfig
from backtests.event_study import run_event_study
from backtests.metrics import max_drawdown, write_json
from backtests.regimes import run_regime_event_study
from backtests.robustness import run_robustness
from backtests.sanity import run_sanity_checks
from backtests.sim_execution import run_execution_simulation
from backtests.walkforward import run_walkforward
from bot.pattern_mining.dataset_builder import build_synchronized_dataset
from bot.pattern_mining.feature_engineering import PATTERN_FEATURE_COLUMNS
from bot.pattern_mining.payoff_model import PayoffPredictor, train_payoff_models


TIMESTAMP_CANDIDATES = ["open_time", "timestamp", "ts", "datetime", "date"]


def _load_csv(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return df

    lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    for required in ["open", "high", "low", "close", "volume"]:
        if required in lower_map and lower_map[required] != required:
            rename[lower_map[required]] = required
    if rename:
        df = df.rename(columns=rename)

    ts_col = next((c for c in TIMESTAMP_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"Timestamp column not found in {path}")
    if ts_col != "open_time":
        df = df.rename(columns={ts_col: "open_time"})

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.sort_values("open_time").reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run offline validation pipeline for pattern mining")
    p.add_argument("--ohlcv", required=True, help="Path to OHLCV CSV")
    p.add_argument("--oi", default=None, help="Path to OI CSV")
    p.add_argument("--cvd", default=None, help="Path to CVD CSV")
    p.add_argument("--liq", default=None, help="Path to Liquidations CSV")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--output_dir", default="reports")

    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--wick", type=float, default=None)
    p.add_argument("--min_support", type=int, default=None)
    p.add_argument("--sweep_lookback", type=int, default=None)
    p.add_argument("--train_window", type=int, default=None)
    p.add_argument("--test_window", type=int, default=None)
    p.add_argument("--slippage_bps", type=float, default=None)
    p.add_argument("--fee_bps", type=float, default=None)
    p.add_argument("--cooldown", type=int, default=None)
    p.add_argument("--max_trades_per_day", type=int, default=None)
    p.add_argument("--payoff_ratio_min", type=float, default=None)
    p.add_argument("--payoff_expected_min", type=float, default=None)
    p.add_argument("--payoff_train_split", type=float, default=None)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--compare_before_after", action="store_true")
    p.add_argument("--before_config", action="append", default=[])
    p.add_argument("--after_config", action="append", default=[])
    return p.parse_args()


def _build_config(args: argparse.Namespace, run_output_dir: Path) -> ValidationConfig:
    cfg = ValidationConfig(timeframe=args.timeframe, output_dir=str(run_output_dir), seed=int(args.seed))
    if args.horizon is not None:
        cfg.horizon_candles = int(args.horizon)
    if args.target is not None:
        cfg.target_return = float(args.target)
    if args.wick is not None:
        cfg.wick_threshold = float(args.wick)
    if args.min_support is not None:
        cfg.min_support = int(args.min_support)
    if args.sweep_lookback is not None:
        cfg.sweep_lookback = int(args.sweep_lookback)
    if args.train_window is not None:
        cfg.train_window_candles = int(args.train_window)
    if args.test_window is not None:
        cfg.test_window_candles = int(args.test_window)
    if args.slippage_bps is not None:
        cfg.slippage_bps = float(args.slippage_bps)
    if args.fee_bps is not None:
        cfg.fee_bps = float(args.fee_bps)
    if args.cooldown is not None:
        cfg.cooldown_candles = int(args.cooldown)
    if args.max_trades_per_day is not None:
        cfg.max_trades_per_day = int(args.max_trades_per_day)
    if args.payoff_ratio_min is not None:
        cfg.payoff_ratio_min = float(args.payoff_ratio_min)
    if args.payoff_expected_min is not None:
        cfg.payoff_expected_min = float(args.payoff_expected_min)
    if args.payoff_train_split is not None:
        cfg.payoff_train_split = float(args.payoff_train_split)
    cfg.payoff_horizon = int(cfg.horizon_candles)
    return cfg


def _parse_scalar(v: str) -> Any:
    lower = v.strip().lower()
    if lower in {"1", "true", "yes", "on"}:
        return True
    if lower in {"0", "false", "no", "off"}:
        return False
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v


def _apply_overrides(cfg: ValidationConfig, overrides: list[str]) -> ValidationConfig:
    alias = {
        "regime_adjustments": "enable_regime_adjustments",
        "payoff_filter": "enable_payoff_filter",
    }
    out = ValidationConfig(**cfg.__dict__)
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        key = alias.get(key, key)
        if not hasattr(out, key):
            continue
        parsed = _parse_scalar(value)
        current = getattr(out, key)
        if isinstance(current, bool):
            parsed = bool(parsed)
        elif isinstance(current, int) and not isinstance(current, bool):
            parsed = int(parsed)
        elif isinstance(current, float):
            parsed = float(parsed)
        else:
            parsed = str(parsed)
        setattr(out, key, parsed)
    return out


def _run_stage_6_7(
    data: pd.DataFrame,
    cfg: ValidationConfig,
    prepared_df: pd.DataFrame | None = None,
    payoff_predictor: PayoffPredictor | None = None,
) -> dict:
    prepared = prepared_df.copy() if prepared_df is not None else prepare_pattern_frame(data, cfg)
    signals = generate_baseline_signals(prepared, cfg)
    execution = run_execution_simulation(
        data,
        cfg,
        prepared_df=prepared,
        signals=signals,
        payoff_predictor=payoff_predictor,
        persist=True,
    )
    robustness = run_robustness(data, cfg, payoff_predictor=payoff_predictor)
    return {
        "execution": execution,
        "robustness": robustness,
    }


def _train_payoff_for_after(
    data: pd.DataFrame,
    cfg: ValidationConfig,
    after_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, PayoffPredictor | None, dict[str, Any]]:
    prepared = prepare_pattern_frame(data, cfg)
    split_idx = int(len(prepared) * float(cfg.payoff_train_split))
    split_idx = max(200, min(split_idx, len(prepared) - 200)) if len(prepared) > 500 else max(1, len(prepared) // 2)

    train_df = prepared.iloc[:split_idx].copy()
    test_df = prepared.iloc[split_idx:].copy()

    info: dict[str, Any] = {
        "train_start": str(train_df["open_time"].iloc[0]) if not train_df.empty else None,
        "train_end": str(train_df["open_time"].iloc[-1]) if not train_df.empty else None,
        "test_start": str(test_df["open_time"].iloc[0]) if not test_df.empty else None,
        "test_end": str(test_df["open_time"].iloc[-1]) if not test_df.empty else None,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "status": "disabled",
    }

    feature_cols = [c for c in PATTERN_FEATURE_COLUMNS if c in train_df.columns]
    predictor: PayoffPredictor | None = None
    models_dir = after_dir / "models"
    metrics_path = after_dir / "payoff_model_metrics.json"
    fi_path = after_dir / "payoff_feature_importance.csv"
    sample_path = after_dir / "payoff_predictions_sample.csv"

    try:
        artifacts = train_payoff_models(train_df, feature_cols, cfg, model_dir=models_dir)
        predictor = PayoffPredictor(artifacts)
        info["status"] = "trained"

        metrics_df = artifacts.metrics.copy()
        avg_metrics = (
            metrics_df.groupby("target")[["mae", "rmse", "r2"]].mean().reset_index().to_dict(orient="records")
            if not metrics_df.empty
            else []
        )
        write_json(
            metrics_path,
            {
                "splits": metrics_df.to_dict(orient="records"),
                "avg_by_target": avg_metrics,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
            },
        )
        artifacts.feature_importance.to_csv(fi_path, index=False)

        pred_rows = []
        runup_col = artifacts.target_runup_col
        ddown_col = artifacts.target_ddown_col
        for _, row in test_df.head(200).iterrows():
            payload = row.to_dict()
            pr, pdw = predictor.predict_payoff(payload)
            pred_rows.append(
                {
                    "open_time": row["open_time"],
                    "pred_runup": pr,
                    "pred_ddown_abs": pdw,
                    "true_runup": float(row.get(runup_col, np.nan)),
                    "true_ddown_abs": abs(float(row.get(ddown_col, np.nan))),
                }
            )
        pd.DataFrame(pred_rows).to_csv(sample_path, index=False)

        # Precompute predictions for the full test slice for faster stage 6/7.
        preds_runup: list[float | None] = []
        preds_dd: list[float | None] = []
        for _, row in test_df.iterrows():
            pr, pdw = predictor.predict_payoff(row.to_dict())
            preds_runup.append(pr)
            preds_dd.append(pdw)
        test_df["pred_runup"] = preds_runup
        test_df["pred_ddown_abs"] = preds_dd
    except Exception as exc:
        info["status"] = "disabled_insufficient_data"
        info["reason"] = str(exc)
        write_json(metrics_path, {"status": info["status"], "reason": str(exc)})
        pd.DataFrame(columns=["feature", "importance_runup", "importance_ddown", "importance_mean"]).to_csv(fi_path, index=False)
        pd.DataFrame(columns=["open_time", "pred_runup", "pred_ddown_abs", "true_runup", "true_ddown_abs"]).to_csv(
            sample_path, index=False
        )
        predictor = None

    return train_df, test_df, predictor, info


def _equity_stats(equity_curve: pd.DataFrame) -> dict[str, float]:
    if equity_curve.empty or "equity" not in equity_curve.columns:
        return {"equity_final_return": 0.0, "equity_max_drawdown": 0.0}
    eq = pd.to_numeric(equity_curve["equity"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if eq.empty:
        return {"equity_final_return": 0.0, "equity_max_drawdown": 0.0}
    return {
        "equity_final_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0) if eq.iloc[0] != 0 else 0.0,
        "equity_max_drawdown": float(max_drawdown(eq)),
    }


def _top5_with_filter(grid: pd.DataFrame, max_dd_filter: float = -0.20) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    filt = grid[grid["execution_max_dd"] >= max_dd_filter]
    if filt.empty:
        filt = grid
    return filt.sort_values(["execution_expectancy", "execution_max_dd"], ascending=[False, False]).head(5)


def _build_comparison(before: dict, after: dict, training_info: dict[str, Any] | None = None) -> tuple[dict, pd.DataFrame]:
    b_exec = before["execution"]["metrics"]
    a_exec = after["execution"]["metrics"]

    b_grid = before["robustness"]["grid"]
    a_grid = after["robustness"]["grid"]

    b_equity = _equity_stats(before["execution"]["equity_curve"])
    a_equity = _equity_stats(after["execution"]["equity_curve"])

    rows = []

    exec_keys = ["expectancy", "winrate", "profit_factor", "max_drawdown", "trades", "avg_hold"]
    for key in exec_keys:
        bv = float(b_exec.get(key, 0.0))
        av = float(a_exec.get(key, 0.0))
        rows.append({"group": "execution", "metric": key, "before": bv, "after": av, "delta": av - bv})

    rows.append(
        {
            "group": "robustness",
            "metric": "best_execution_expectancy",
            "before": float(b_grid["execution_expectancy"].max()) if not b_grid.empty else 0.0,
            "after": float(a_grid["execution_expectancy"].max()) if not a_grid.empty else 0.0,
            "delta": (float(a_grid["execution_expectancy"].max()) if not a_grid.empty else 0.0)
            - (float(b_grid["execution_expectancy"].max()) if not b_grid.empty else 0.0),
        }
    )
    rows.append(
        {
            "group": "robustness",
            "metric": "best_execution_max_dd",
            "before": float(b_grid["execution_max_dd"].max()) if not b_grid.empty else 0.0,
            "after": float(a_grid["execution_max_dd"].max()) if not a_grid.empty else 0.0,
            "delta": (float(a_grid["execution_max_dd"].max()) if not a_grid.empty else 0.0)
            - (float(b_grid["execution_max_dd"].max()) if not b_grid.empty else 0.0),
        }
    )

    for key in ["blocked_edge_total", "blocked_by_regime_rule_total", "blocked_by_payoff_total"]:
        bv = float(b_exec.get(key, 0.0))
        av = float(a_exec.get(key, 0.0))
        rows.append({"group": "blocked", "metric": key, "before": bv, "after": av, "delta": av - bv})

    for key in ["equity_final_return", "equity_max_drawdown"]:
        bv = float(b_equity.get(key, 0.0))
        av = float(a_equity.get(key, 0.0))
        rows.append({"group": "equity", "metric": key, "before": bv, "after": av, "delta": av - bv})

    table = pd.DataFrame(rows)

    improvement = (
        float(a_exec.get("expectancy", 0.0)) > float(b_exec.get("expectancy", 0.0))
        and float(a_exec.get("max_drawdown", 0.0)) > float(b_exec.get("max_drawdown", 0.0))
    )

    summary = {
        "improvement": bool(improvement),
        "before": {
            "execution": b_exec,
            "equity": b_equity,
            "robustness_best_execution_expectancy": float(b_grid["execution_expectancy"].max()) if not b_grid.empty else 0.0,
            "robustness_best_execution_max_dd": float(b_grid["execution_max_dd"].max()) if not b_grid.empty else 0.0,
            "robustness_top5": _top5_with_filter(b_grid).to_dict(orient="records"),
        },
        "after": {
            "execution": a_exec,
            "equity": a_equity,
            "robustness_best_execution_expectancy": float(a_grid["execution_expectancy"].max()) if not a_grid.empty else 0.0,
            "robustness_best_execution_max_dd": float(a_grid["execution_max_dd"].max()) if not a_grid.empty else 0.0,
            "robustness_top5": _top5_with_filter(a_grid).to_dict(orient="records"),
        },
        "payoff_training": training_info or {},
    }
    return summary, table


def _print_final_summary(
    baseline_result: dict,
    event_result: dict,
    regime_result: dict,
    walk_result: dict,
    execution_result: dict,
    robustness_result: dict,
) -> None:
    b = baseline_result["metrics"]
    e = execution_result["metrics"]
    print("\nValidation summary")
    print(f"- baseline expectancy={b.get('expectancy', 0.0):.6f} max_dd={b.get('max_drawdown', 0.0):.6f}")

    best_event = event_result.get("best_event")
    if best_event:
        print(
            "- best global event "
            f"{best_event.get('event')} edge={float(best_event.get('edge', 0.0)):.4f} support={int(best_event.get('count', 0))}"
        )
    else:
        print("- best global event: n/a")

    best_by_regime = regime_result.get("best_by_regime", {})
    if best_by_regime:
        print("- best event by regime:")
        for regime_id, payload in best_by_regime.items():
            print(
                f"  {regime_id}: {payload.get('event')} edge={float(payload.get('edge', 0.0)):.4f} "
                f"support={int(payload.get('count', 0))}"
            )
    else:
        print("- best event by regime: n/a")

    print(f"- walk-forward mean AUC={walk_result.get('mean_auc', 0.0):.6f}")
    print(f"- execution expectancy={e.get('expectancy', 0.0):.6f} max_dd={e.get('max_drawdown', 0.0):.6f}")

    top5 = robustness_result["top_configs"].head(5)
    if top5.empty:
        print("- robustness top 5: n/a")
    else:
        print("- robustness top 5 configs:")
        for _, row in top5.iterrows():
            print(
                "  "
                f"wick={row['wick_threshold']:.2f} horizon={int(row['horizon_candles'])} target={row['target_return']:.4f} "
                f"exp={row['execution_expectancy']:.6f} max_dd={row['execution_max_dd']:.6f}"
            )


def main() -> None:
    args = _parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_output_dir = Path(args.output_dir) / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_config(args, run_output_dir)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    ohlcv = _load_csv(args.ohlcv)
    oi = _load_csv(args.oi)
    cvd = _load_csv(args.cvd)
    liq = _load_csv(args.liq)

    if ohlcv is None or ohlcv.empty:
        raise ValueError("OHLCV input is empty")

    data = build_synchronized_dataset(ohlcv=ohlcv, oi=oi, cvd=cvd, liquidations=liq)

    print(f"run_id={run_id}")
    print(f"output_dir={run_output_dir}")

    if args.compare_before_after:
        before_dir = run_output_dir / "before"
        after_dir = run_output_dir / "after"
        before_dir.mkdir(parents=True, exist_ok=True)
        after_dir.mkdir(parents=True, exist_ok=True)

        before_overrides = list(args.before_config)
        after_overrides = list(args.after_config)
        if not before_overrides:
            before_overrides = ["payoff_filter=0"]
        if not after_overrides:
            after_overrides = ["payoff_filter=1"]

        before_cfg = _apply_overrides(cfg, before_overrides)
        after_cfg = _apply_overrides(cfg, after_overrides)
        before_cfg.output_dir = str(before_dir)
        after_cfg.output_dir = str(after_dir)
        before_cfg.enable_payoff_filter = False if "payoff_filter=0" in before_overrides else bool(before_cfg.enable_payoff_filter)

        print("mode=compare_before_after")
        print(f"before_config={before_overrides}")
        print(f"after_config={after_overrides}")

        train_df, test_df, payoff_predictor, training_info = _train_payoff_for_after(data, after_cfg, after_dir)
        data_test = data[data["open_time"] >= test_df["open_time"].min()].copy().reset_index(drop=True) if not test_df.empty else data.copy()
        prepared_test = test_df.copy().reset_index(drop=True) if not test_df.empty else prepare_pattern_frame(data_test, after_cfg)

        if payoff_predictor is None:
            after_cfg.enable_payoff_filter = False
            training_info["status"] = "disabled_insufficient_data"

        before_result = _run_stage_6_7(data_test, before_cfg, prepared_df=prepared_test, payoff_predictor=None)
        after_result = _run_stage_6_7(data_test, after_cfg, prepared_df=prepared_test, payoff_predictor=payoff_predictor)

        summary, table = _build_comparison(before_result, after_result, training_info=training_info)
        write_json(run_output_dir / "comparison_summary.json", summary)
        table.to_csv(run_output_dir / "comparison_table.csv", index=False)

        delta_expectancy = float(summary["after"]["execution"].get("expectancy", 0.0)) - float(
            summary["before"]["execution"].get("expectancy", 0.0)
        )
        delta_max_dd = float(summary["after"]["execution"].get("max_drawdown", 0.0)) - float(
            summary["before"]["execution"].get("max_drawdown", 0.0)
        )
        print(f"IMPROVEMENT={str(summary['improvement']).lower()}")
        print(f"delta_expectancy={delta_expectancy:.6f}")
        print(f"delta_max_drawdown={delta_max_dd:.6f}")
        return

    # Required execution order
    run_sanity_checks(data, cfg)
    baseline_result = run_baseline(data, cfg)
    event_result = run_event_study(data, cfg)
    regime_result = run_regime_event_study(data, cfg)
    walk_result = run_walkforward(data, cfg)
    execution_result = run_execution_simulation(data, cfg)
    robustness_result = run_robustness(data, cfg)

    _print_final_summary(
        baseline_result,
        event_result,
        regime_result,
        walk_result,
        execution_result,
        robustness_result,
    )


if __name__ == "__main__":
    main()
