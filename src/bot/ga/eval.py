from __future__ import annotations

import csv
import io
import json
import math
import re
import subprocess
import sys
import shutil
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bot.ga.space import SearchSpace, genes_hash
from bot.runs.serializers import write_json, write_yaml


@dataclass
class EvalResult:
    generation: int
    index: int
    genes: dict[str, Any]
    genes_hash: str
    fitness: float
    metrics: dict[str, Any]
    run_dir: str
    cached: bool = False
    error: str | None = None


def compute_fitness(
    summary: dict[str, Any],
    objective: str = "score",
    min_trades_hard: int = 30,
    target_trades: int = 140,
    min_trades_for_sharpe: int = 120,
    lambda_trades: float = 6.0,
    w_ret: float = 1.0,
    w_dd: float = 0.6,
    w_sharpe: float = 10.0,
    hard_cut_value: float = -10_000.0,
) -> tuple[float, dict[str, Any], list[str]]:
    warnings: list[str] = []

    def get_metric(keys: list[str], default: float = 0.0) -> float:
        for key in keys:
            value = summary.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue
        warnings.append(f"missing:{'/'.join(keys)}")
        return default

    if objective != "score":
        value = summary.get(objective)
        if value is None:
            warnings.append(f"objective_missing:{objective}")
            return float("-inf"), {}, warnings
        try:
            return float(value), {objective: float(value)}, warnings
        except Exception:
            warnings.append(f"objective_invalid:{objective}")
            return float("-inf"), {}, warnings

    ret = get_metric(["return_net", "cagr"], default=math.nan)
    if math.isnan(ret):
        ret = get_metric(["return_pct", "total_return_pct"], default=0.0) / 100.0

    dd = abs(get_metric(["max_drawdown", "dd"], default=math.nan))
    if math.isnan(dd):
        dd = abs(get_metric(["max_drawdown_pct"], default=0.0)) / 100.0

    sharpe_raw = summary.get("sharpe", summary.get("sharpe_ratio"))
    sharpe: float | None = None
    if sharpe_raw is not None:
        try:
            sharpe = float(sharpe_raw)
        except Exception:
            sharpe = None
            warnings.append("invalid:sharpe")
    else:
        warnings.append("missing:sharpe/sharpe_ratio")

    trades = int(round(get_metric(["total_trades", "trades"], default=0.0)))
    switches = int(round(get_metric(["regime_switch_count_total", "switches_total"], default=0.0)))

    deficit_ratio = max(0.0, (float(target_trades - trades) / max(1.0, float(target_trades))))
    penalty_trades = lambda_trades * (deficit_ratio**2)

    ret_term = w_ret * (100.0 * float(ret))
    dd_term = w_dd * (100.0 * float(dd))
    sharpe_term = 0.0
    if trades >= min_trades_for_sharpe and sharpe is not None:
        sharpe_term = w_sharpe * sharpe

    hard_cut_applied = trades < min_trades_hard
    hard_cut_reason = "invalid_low_trades_hard" if hard_cut_applied else ""
    if hard_cut_applied:
        score = hard_cut_value
    else:
        score = ret_term - dd_term + sharpe_term - penalty_trades

    components = {
        "ret": float(ret),
        "dd": float(dd),
        "sharpe": sharpe,
        "trades": trades,
        "switches": switches,
        "ret_term": float(ret_term),
        "dd_term": float(dd_term),
        "sharpe_term": float(sharpe_term),
        "penalty_trades": float(penalty_trades),
        "min_trades_hard": int(min_trades_hard),
        "target_trades": int(target_trades),
        "min_trades_for_sharpe": int(min_trades_for_sharpe),
        "lambda_trades": float(lambda_trades),
        "w_ret": float(w_ret),
        "w_dd": float(w_dd),
        "w_sharpe": float(w_sharpe),
        "hard_cut_applied": bool(hard_cut_applied),
        "hard_cut_reason": hard_cut_reason,
        "invalid_low_trades_hard": bool(hard_cut_applied),
        "fitness_total": float(score),
        "score": float(score),
    }
    if hard_cut_applied:
        warnings.append("viability:invalid_low_trades_hard")
    return float(score), components, warnings


def _parse_metrics_md(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    if not path.exists():
        return parsed

    line_re = re.compile(r"^- \*\*(?P<k>[^*]+)\*\*: (?P<v>.+)$")
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            match = line_re.match(line)
            if not match:
                continue
            key = match.group("k").strip()
            value_str = match.group("v").strip()
            try:
                parsed[key] = float(value_str)
            except ValueError:
                if value_str.lower() in {"true", "false"}:
                    parsed[key] = value_str.lower() == "true"
                else:
                    parsed[key] = value_str
    return parsed


def _parse_equity_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    values: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                values.append(float(row.get("equity", "")))
            except Exception:
                continue
    if len(values) < 2:
        return {}

    first = values[0]
    last = values[-1]
    peak = first
    max_dd = 0.0
    for val in values:
        peak = max(peak, val)
        if peak > 0:
            max_dd = min(max_dd, (val / peak) - 1.0)

    return {
        "return_net": (last / first) - 1.0 if first > 0 else 0.0,
        "max_drawdown": max_dd,
    }


def load_summary_with_fallback(run_dir: Path) -> tuple[dict[str, Any], str]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f), "summary.json"

    metrics = _parse_metrics_md(run_dir / "metrics.md")
    if metrics:
        return metrics, "metrics.md"

    equity = _parse_equity_csv(run_dir / "equity.csv")
    if equity:
        return equity, "equity.csv"

    return {}, "none"


def _failure_payload(
    *,
    objective: str,
    min_trades_hard: int,
    target_trades: int,
    min_trades_for_sharpe: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "fitness": float("-inf"),
        "objective": objective,
        "fitness_components": {
            "min_trades_hard": int(min_trades_hard),
            "target_trades": int(target_trades),
            "min_trades_for_sharpe": int(min_trades_for_sharpe),
            "ret_term": 0.0,
            "dd_term": 0.0,
            "sharpe_term": 0.0,
            "penalty_trades": 0.0,
            "hard_cut_applied": True,
            "hard_cut_reason": "backtest_failed",
            "invalid_low_trades_hard": False,
            "fitness_total": float("-inf"),
            "score": float("-inf"),
        },
        "warnings": [reason],
        "metrics_source": "none",
        "summary": {},
    }


def evaluate_candidate(
    *,
    generation: int,
    index: int,
    genes: dict[str, Any],
    space: SearchSpace,
    outdir: Path,
    config_path: str,
    data_path: str,
    funding_path: str | None,
    objective: str,
    min_trades_hard: int,
    target_trades: int,
    min_trades_for_sharpe: int,
    lambda_trades: float,
    w_ret: float,
    w_dd: float,
    w_sharpe: float,
    eval_backend: str = "inprocess",
    save_full_artifacts: bool = False,
) -> EvalResult:
    ind_hash = genes_hash(genes)
    gen_dir = outdir / f"gen_{generation:05d}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"ind_{index:04d}_{ind_hash[:10]}"
    run_dir = gen_dir / run_name

    candidate_cfg = space.apply_genes(genes)
    candidate_cfg_path: Path | None = None
    if eval_backend == "subprocess":
        candidate_cfg_path = gen_dir / f"candidate_{run_name}.yaml"
        write_yaml(candidate_cfg_path, candidate_cfg)

    existing_summary = run_dir / "summary.json"
    existing_eval = run_dir / "ga_eval.json"
    if existing_summary.exists():
        summary, source = load_summary_with_fallback(run_dir)
        fitness, components, warnings = compute_fitness(
            summary,
            objective=objective,
            min_trades_hard=min_trades_hard,
            target_trades=target_trades,
            min_trades_for_sharpe=min_trades_for_sharpe,
            lambda_trades=lambda_trades,
            w_ret=w_ret,
            w_dd=w_dd,
            w_sharpe=w_sharpe,
        )
        payload = {
            "fitness": fitness,
            "objective": objective,
            "fitness_components": components,
            "warnings": warnings,
            "metrics_source": source,
            "summary": summary,
        }
        if not existing_eval.exists():
            write_json(existing_eval, payload)
        return EvalResult(
            generation=generation,
            index=index,
            genes=genes,
            genes_hash=ind_hash,
            fitness=fitness,
            metrics=payload,
            run_dir=str(run_dir),
            cached=True,
        )

    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    if eval_backend == "inprocess":
        return _evaluate_candidate_inprocess(
            generation=generation,
            index=index,
            genes=genes,
            config_path=config_path,
            data_path=data_path,
            funding_path=funding_path,
            objective=objective,
            min_trades_hard=min_trades_hard,
            target_trades=target_trades,
            min_trades_for_sharpe=min_trades_for_sharpe,
            lambda_trades=lambda_trades,
            w_ret=w_ret,
            w_dd=w_dd,
            w_sharpe=w_sharpe,
            run_dir=run_dir,
            candidate_cfg=candidate_cfg,
            ind_hash=ind_hash,
            save_full_artifacts=save_full_artifacts,
        )

    assert candidate_cfg_path is not None
    return _evaluate_candidate_subprocess(
        generation=generation,
        index=index,
        genes=genes,
        config_path=config_path,
        data_path=data_path,
        funding_path=funding_path,
        objective=objective,
        min_trades_hard=min_trades_hard,
        target_trades=target_trades,
        min_trades_for_sharpe=min_trades_for_sharpe,
        lambda_trades=lambda_trades,
        w_ret=w_ret,
        w_dd=w_dd,
        w_sharpe=w_sharpe,
        run_dir=run_dir,
        candidate_cfg=candidate_cfg,
        candidate_cfg_path=candidate_cfg_path,
        ind_hash=ind_hash,
        run_name=run_name,
        gen_dir=gen_dir,
        save_full_artifacts=save_full_artifacts,
    )


def _evaluate_candidate_subprocess(
    *,
    generation: int,
    index: int,
    genes: dict[str, Any],
    config_path: str,
    data_path: str,
    funding_path: str | None,
    objective: str,
    min_trades_hard: int,
    target_trades: int,
    min_trades_for_sharpe: int,
    lambda_trades: float,
    w_ret: float,
    w_dd: float,
    w_sharpe: float,
    run_dir: Path,
    candidate_cfg: dict[str, Any],
    candidate_cfg_path: Path,
    ind_hash: str,
    run_name: str,
    gen_dir: Path,
    save_full_artifacts: bool,
) -> EvalResult:
    del config_path
    cmd = [
        sys.executable,
        "-m",
        "bot",
        "backtest",
        "--data-path",
        str(data_path),
        "--config",
        str(candidate_cfg_path),
        "--outdir",
        str(gen_dir),
        "--run-name",
        run_name,
        "--tag",
        f"ga_gen_{generation}",
    ]
    if funding_path:
        cmd.extend(["--funding-path", str(funding_path)])

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "backtest.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (run_dir / "backtest.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    write_yaml(run_dir / "candidate_config.yaml", candidate_cfg)
    if completed.returncode != 0:
        payload = _failure_payload(
            objective=objective,
            min_trades_hard=min_trades_hard,
            target_trades=target_trades,
            min_trades_for_sharpe=min_trades_for_sharpe,
            reason=f"backtest_failed:{completed.returncode}",
        )
        write_json(run_dir / "ga_eval.json", payload)
        return EvalResult(
            generation=generation,
            index=index,
            genes=genes,
            genes_hash=ind_hash,
            fitness=float("-inf"),
            metrics=payload,
            run_dir=str(run_dir),
            error=f"backtest_failed:{completed.returncode}",
        )

    summary, source = load_summary_with_fallback(run_dir)
    if not summary:
        payload = _failure_payload(
            objective=objective,
            min_trades_hard=min_trades_hard,
            target_trades=target_trades,
            min_trades_for_sharpe=min_trades_for_sharpe,
            reason="missing_metrics",
        )
        write_json(run_dir / "ga_eval.json", payload)
        return EvalResult(
            generation=generation,
            index=index,
            genes=genes,
            genes_hash=ind_hash,
            fitness=float("-inf"),
            metrics=payload,
            run_dir=str(run_dir),
            error="missing_metrics",
        )

    fitness, components, warnings = compute_fitness(
        summary,
        objective=objective,
        min_trades_hard=min_trades_hard,
        target_trades=target_trades,
        min_trades_for_sharpe=min_trades_for_sharpe,
        lambda_trades=lambda_trades,
        w_ret=w_ret,
        w_dd=w_dd,
        w_sharpe=w_sharpe,
    )
    payload = {
        "fitness": fitness,
        "objective": objective,
        "fitness_components": components,
        "warnings": warnings,
        "metrics_source": source,
        "summary": summary,
    }
    write_json(run_dir / "ga_eval.json", payload)
    if not save_full_artifacts:
        for filename in ["trades.csv", "equity.csv", "metrics.md", "regime_stats.json", "direction_stats.json"]:
            (run_dir / filename).unlink(missing_ok=True)

    return EvalResult(
        generation=generation,
        index=index,
        genes=genes,
        genes_hash=ind_hash,
        fitness=fitness,
        metrics=payload,
        run_dir=str(run_dir),
    )


def _evaluate_candidate_inprocess(
    *,
    generation: int,
    index: int,
    genes: dict[str, Any],
    config_path: str,
    data_path: str,
    funding_path: str | None,
    objective: str,
    min_trades_hard: int,
    target_trades: int,
    min_trades_for_sharpe: int,
    lambda_trades: float,
    w_ret: float,
    w_dd: float,
    w_sharpe: float,
    run_dir: Path,
    candidate_cfg: dict[str, Any],
    ind_hash: str,
    save_full_artifacts: bool,
) -> EvalResult:
    from bot.cli import _execute_backtest
    from bot.runs.run_manager import RunManager
    from bot.utils.config import Settings

    run_name = run_dir.name
    gen_dir = run_dir.parent
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        run_cfg = Settings.model_validate(candidate_cfg)
        if save_full_artifacts:
            manager = RunManager(gen_dir)
            ctx = manager.build_context(run_cfg, run_name=run_name)
            manager.init_run(
                ctx,
                settings=run_cfg,
                data_path=data_path,
                config_path=config_path,
                seed=None,
                tag=f"ga_gen_{generation}",
            )
            write_yaml(run_dir / "candidate_config.yaml", candidate_cfg)
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                trades, equity, summary, regime_stats, direction_stats, ms_stats, pivots = _execute_backtest(
                    data_path,
                    run_cfg,
                    funding_path=funding_path,
                )
                manager.persist_outputs(
                    ctx,
                    summary=summary,
                    trades=trades,
                    equity=equity,
                    regime_stats=regime_stats,
                    direction_stats=direction_stats,
                    market_structure_stats=ms_stats,
                    pivots=pivots,
                )
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_yaml(run_dir / "candidate_config.yaml", candidate_cfg)
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                _, _, summary, _, _, _, _ = _execute_backtest(
                    data_path,
                    run_cfg,
                    funding_path=funding_path,
                )
                write_json(run_dir / "summary.json", summary)
    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "backtest.stdout.log").write_text(stdout_capture.getvalue(), encoding="utf-8")
        (run_dir / "backtest.stderr.log").write_text(
            stderr_capture.getvalue() + "\n" + traceback.format_exc(),
            encoding="utf-8",
        )
        payload = _failure_payload(
            objective=objective,
            min_trades_hard=min_trades_hard,
            target_trades=target_trades,
            min_trades_for_sharpe=min_trades_for_sharpe,
            reason=f"backtest_failed:{type(exc).__name__}",
        )
        write_json(run_dir / "ga_eval.json", payload)
        return EvalResult(
            generation=generation,
            index=index,
            genes=genes,
            genes_hash=ind_hash,
            fitness=float("-inf"),
            metrics=payload,
            run_dir=str(run_dir),
            error=f"inprocess_failed:{type(exc).__name__}",
        )

    if save_full_artifacts:
        (run_dir / "backtest.stdout.log").write_text(stdout_capture.getvalue(), encoding="utf-8")
        (run_dir / "backtest.stderr.log").write_text(stderr_capture.getvalue(), encoding="utf-8")

    summary, source = load_summary_with_fallback(run_dir)
    if not summary:
        payload = _failure_payload(
            objective=objective,
            min_trades_hard=min_trades_hard,
            target_trades=target_trades,
            min_trades_for_sharpe=min_trades_for_sharpe,
            reason="missing_metrics",
        )
        write_json(run_dir / "ga_eval.json", payload)
        return EvalResult(
            generation=generation,
            index=index,
            genes=genes,
            genes_hash=ind_hash,
            fitness=float("-inf"),
            metrics=payload,
            run_dir=str(run_dir),
            error="missing_metrics",
        )

    fitness, components, warnings = compute_fitness(
        summary,
        objective=objective,
        min_trades_hard=min_trades_hard,
        target_trades=target_trades,
        min_trades_for_sharpe=min_trades_for_sharpe,
        lambda_trades=lambda_trades,
        w_ret=w_ret,
        w_dd=w_dd,
        w_sharpe=w_sharpe,
    )
    payload = {
        "fitness": fitness,
        "objective": objective,
        "fitness_components": components,
        "warnings": warnings,
        "metrics_source": source,
        "summary": summary,
    }
    write_json(run_dir / "ga_eval.json", payload)

    return EvalResult(
        generation=generation,
        index=index,
        genes=genes,
        genes_hash=ind_hash,
        fitness=fitness,
        metrics=payload,
        run_dir=str(run_dir),
    )


def make_cache_entry(result: EvalResult) -> dict[str, Any]:
    return {
        "genes_hash": result.genes_hash,
        "fitness": result.fitness,
        "metrics": result.metrics,
        "run_dir": result.run_dir,
        "error": result.error,
    }


def load_cache_entry(generation: int, index: int, genes: dict[str, Any], cache_entry: dict[str, Any]) -> EvalResult:
    return EvalResult(
        generation=generation,
        index=index,
        genes=genes,
        genes_hash=str(cache_entry.get("genes_hash") or genes_hash(genes)),
        fitness=float(cache_entry.get("fitness", float("-inf"))),
        metrics=dict(cache_entry.get("metrics", {})),
        run_dir=str(cache_entry.get("run_dir", "")),
        cached=True,
        error=cache_entry.get("error"),
    )
