from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from backtests.baseline import generate_baseline_signals, prepare_pattern_frame, run_baseline
from backtests.config import ValidationConfig
from backtests.sim_execution import run_execution_simulation
from bot.pattern_mining.payoff_model import PayoffPredictor


def run_robustness(df: pd.DataFrame, cfg: ValidationConfig, payoff_predictor: PayoffPredictor | None = None) -> dict:
    if cfg.enable_payoff_filter:
        wick_values = [0.5, 0.7]
        horizon_values = [3, 12]
        target_values = [0.0045, 0.009]
        payoff_ratio_values = [1.0, 1.1, 1.2, 1.3, 1.5]
        payoff_expected_values = [-0.0002, 0.0, 0.0002, 0.0005]
    else:
        wick_values = [0.4, 0.5, 0.6, 0.7, 0.8]
        horizon_values = [2, 3, 4, 6, 8, 12]
        target_values = [0.003, 0.0045, 0.006, 0.009, 0.012]
        payoff_ratio_values = [cfg.payoff_ratio_min]
        payoff_expected_values = [cfg.payoff_expected_min]

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[tuple[float, int], tuple[pd.DataFrame, pd.Series, dict]] = {}
    rows: list[dict] = []

    for wick, horizon, target, payoff_ratio_min, payoff_expected_min in product(
        wick_values, horizon_values, target_values, payoff_ratio_values, payoff_expected_values
    ):
        key = (wick, horizon)

        local_cfg = ValidationConfig(**cfg.__dict__)
        local_cfg.wick_threshold = wick
        local_cfg.horizon_candles = horizon
        local_cfg.payoff_horizon = local_cfg.horizon_candles
        local_cfg.target_return = target
        local_cfg.payoff_ratio_min = float(payoff_ratio_min)
        local_cfg.payoff_expected_min = float(payoff_expected_min)

        if key not in cache:
            prepared = prepare_pattern_frame(df, local_cfg)
            if local_cfg.enable_payoff_filter and payoff_predictor is not None:
                preds_runup: list[float | None] = []
                preds_dd: list[float | None] = []
                for _, row in prepared.iterrows():
                    pr, pdw = payoff_predictor.predict_payoff(row.to_dict())
                    preds_runup.append(pr)
                    preds_dd.append(pdw)
                prepared["pred_runup"] = preds_runup
                prepared["pred_ddown_abs"] = preds_dd
            signals = generate_baseline_signals(prepared, local_cfg)
            baseline = run_baseline(df, local_cfg, prepared_df=prepared, signals=signals, persist=False)
            cache[key] = (prepared, signals, baseline["metrics"])

        prepared_df, signals, baseline_metrics = cache[key]
        exec_result = run_execution_simulation(
            df,
            local_cfg,
            prepared_df=prepared_df,
            signals=signals,
            payoff_predictor=payoff_predictor,
            persist=False,
        )

        rows.append(
            {
                "wick_threshold": wick,
                "horizon_candles": horizon,
                "target_return": target,
                "payoff_ratio_min": float(payoff_ratio_min),
                "payoff_expected_min": float(payoff_expected_min),
                "baseline_expectancy": float(baseline_metrics.get("expectancy", 0.0)),
                "baseline_max_dd": float(baseline_metrics.get("max_drawdown", 0.0)),
                "execution_expectancy": float(exec_result["metrics"].get("expectancy", 0.0)),
                "execution_max_dd": float(exec_result["metrics"].get("max_drawdown", 0.0)),
                "execution_trades": int(exec_result["metrics"].get("trades", 0)),
                "blocked_edge_total": int(exec_result["metrics"].get("blocked_edge_total", 0)),
                "blocked_by_regime_rule_total": int(exec_result["metrics"].get("blocked_by_regime_rule_total", 0)),
                "blocked_by_payoff_total": int(exec_result["metrics"].get("blocked_by_payoff_total", 0)),
            }
        )

    grid = pd.DataFrame(rows).sort_values(
        ["execution_expectancy", "execution_max_dd"],
        ascending=[False, False],
    )
    grid.to_csv(out_dir / "robustness_grid.csv", index=False)

    max_dd_filter = -0.20
    top = grid[grid["execution_max_dd"] >= max_dd_filter].head(20).copy()
    if top.empty:
        top = grid.head(20).copy()
    top.to_csv(out_dir / "robustness_top_configs.csv", index=False)

    return {
        "grid": grid,
        "top_configs": top,
    }
