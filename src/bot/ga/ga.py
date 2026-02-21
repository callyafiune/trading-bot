from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

from rich import print

from bot.ga.eval import EvalResult, evaluate_candidate, load_cache_entry, make_cache_entry
from bot.ga.space import SearchSpace, discover_search_space, genes_hash
from bot.runs.serializers import write_json, write_yaml

HOF_SIZE = 50
_WORKER_CTX: dict[str, Any] = {}


@dataclass
class GASettings:
    data_path: str
    funding_path: str | None
    config_path: str
    outdir: str
    population: int
    elite: int
    tournament: int
    cx_prob: float
    mut_prob: float
    seed: int
    n_jobs: int
    resume: bool
    fitness_objective: str
    max_generations: int | None
    max_evals_per_gen: int | None
    print_every: int
    save_best_every: int
    min_trades_hard: int = 30
    target_trades: int = 140
    min_trades_for_sharpe: int = 120
    lambda_trades: float = 6.0
    w_ret: float = 1.0
    w_dd: float = 0.6
    w_sharpe: float = 10.0
    init_baseline_ratio: float = 0.7
    init_baseline_seed_mode: str = "baseline"
    ga_space_path: str | None = None
    eval_backend: str = "inprocess"
    save_full_artifacts: bool = False


@dataclass
class GAState:
    generation: int
    population: list[dict[str, Any]]
    hall_of_fame: list[dict[str, Any]]
    best_global: dict[str, Any] | None
    last_generation_report: dict[str, Any] | None = None
    zero_trade_streak: int = 0


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f) or {}
    if not isinstance(parsed, dict):
        raise ValueError("Config YAML inválido para GA")
    return parsed


def _detect_latest_generation(outdir: Path) -> int | None:
    pattern = re.compile(r"^gen_(\d+)$")
    latest: int | None = None
    for child in outdir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        value = int(match.group(1))
        latest = value if latest is None else max(latest, value)
    return latest


def _clear_console() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _tournament_pick(rng: random.Random, ranked: list[EvalResult], k: int) -> dict[str, Any]:
    k = max(2, min(k, len(ranked)))
    sampled = rng.sample(ranked, k)
    sampled.sort(key=lambda item: item.fitness, reverse=True)
    return dict(sampled[0].genes)


def _uniform_crossover(rng: random.Random, p1: dict[str, Any], p2: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    child: dict[str, Any] = {}
    for key in keys:
        child[key] = p1[key] if rng.random() < 0.5 else p2[key]
    return child


def _mutate_individual(rng: random.Random, child: dict[str, Any], space: SearchSpace, mut_prob: float) -> dict[str, Any]:
    mutated = dict(child)
    if rng.random() >= mut_prob:
        return mutated

    keys = list(space.specs.keys())
    n_mut = max(1, int(math.ceil(len(keys) * 0.1)))
    for key in rng.sample(keys, min(len(keys), n_mut)):
        mutated[key] = space.mutate_gene(key, mutated[key], rng)
    return space.normalize_genes(mutated)


HARD_FILTER_KEYS = [
    "funding_filter.enabled",
    "market_structure.enabled",
    "market_structure.gate.enabled",
    "market_structure.msb.enabled",
    "strategy_breakout.use_ma200_filter",
    "strategy_breakout.use_rel_volume_filter",
    "multi_timeframe.enabled",
    "router.enabled",
]


def _hard_filter_bits(genes: dict[str, Any]) -> dict[str, int]:
    return {
        "funding": 1 if bool(genes.get("funding_filter.enabled", False)) else 0,
        "ms": 1 if bool(genes.get("market_structure.enabled", False)) else 0,
        "gate": 1 if bool(genes.get("market_structure.gate.enabled", False)) else 0,
        "msb": 1 if bool(genes.get("market_structure.msb.enabled", False)) else 0,
        "ma200": 1 if bool(genes.get("strategy_breakout.use_ma200_filter", False)) else 0,
        "relvol": 1 if bool(genes.get("strategy_breakout.use_rel_volume_filter", False)) else 0,
        "router": 1 if bool(genes.get("router.enabled", False)) else 0,
        "mtf": 1 if bool(genes.get("multi_timeframe.enabled", False)) else 0,
    }


def _count_hard_filters(genes: dict[str, Any]) -> int:
    return sum(_hard_filter_bits(genes).values())


def _sanitize_individual(genes: dict[str, Any], space: SearchSpace, rng: random.Random) -> dict[str, Any]:
    fixed = dict(space.normalize_genes(genes))

    # 1) ADX ordering coherence: exit <= enter <= trend.
    adx_keys = ("regime.adx_exit_threshold", "regime.adx_enter_threshold", "regime.adx_trend_threshold")
    if all(k in fixed for k in adx_keys):
        exit_v = float(fixed["regime.adx_exit_threshold"])
        enter_v = float(fixed["regime.adx_enter_threshold"])
        trend_v = float(fixed["regime.adx_trend_threshold"])
        ordered = sorted([exit_v, enter_v, trend_v])
        fixed["regime.adx_exit_threshold"] = ordered[0]
        fixed["regime.adx_enter_threshold"] = ordered[1]
        fixed["regime.adx_trend_threshold"] = ordered[2]

    # 4) Clamp pivot windows.
    if "market_structure.left_bars" in fixed:
        fixed["market_structure.left_bars"] = max(2, min(8, int(fixed["market_structure.left_bars"])))
    if "market_structure.right_bars" in fixed:
        fixed["market_structure.right_bars"] = max(2, min(8, int(fixed["market_structure.right_bars"])))

    # 3) Avoid restrictive hybrid gate setup.
    if fixed.get("market_structure.gate.mode") == "hybrid":
        if "market_structure.gate.hybrid_require_both" in fixed:
            fixed["market_structure.gate.hybrid_require_both"] = False
        if "market_structure.gate.block_in_neutral" in fixed:
            fixed["market_structure.gate.block_in_neutral"] = False

    # 5) Coherence when market_structure is disabled.
    if not bool(fixed.get("market_structure.enabled", False)):
        if "market_structure.gate.enabled" in fixed:
            fixed["market_structure.gate.enabled"] = False
        if "market_structure.msb.enabled" in fixed:
            fixed["market_structure.msb.enabled"] = False

    # 2) Limit active hard filters to at most 2.
    active = [k for k in HARD_FILTER_KEYS if bool(fixed.get(k, False))]
    if len(active) > 2:
        preferred_group = [
            "strategy_breakout.use_ma200_filter",
            "funding_filter.enabled",
            "market_structure.enabled",
        ]
        keep: set[str] = set()
        preferred_active = [k for k in preferred_group if k in active]
        if preferred_active:
            keep.add(rng.choice(preferred_active))

        others = [k for k in active if k not in keep]
        rng.shuffle(others)
        while len(keep) < 2 and others:
            keep.add(others.pop(0))
        for key in active:
            fixed[key] = key in keep

    if not bool(fixed.get("market_structure.enabled", False)):
        if "market_structure.gate.enabled" in fixed:
            fixed["market_structure.gate.enabled"] = False
        if "market_structure.msb.enabled" in fixed:
            fixed["market_structure.msb.enabled"] = False

    return space.normalize_genes(fixed)


def _make_seed_individual(base_genes: dict[str, Any], space: SearchSpace, rng: random.Random, mode: str = "baseline") -> dict[str, Any]:
    genes = space.sample_individual(rng)
    genes.update(base_genes)
    if mode == "baseline":
        overrides: dict[str, Any] = {
            "funding_filter.enabled": False,
            "market_structure.enabled": False,
            "market_structure.gate.enabled": False,
            "market_structure.msb.enabled": False,
            "strategy_breakout.use_rel_volume_filter": False,
            "router.enabled": False,
        }
        for key, value in overrides.items():
            if key in space.specs:
                genes[key] = value
        if "strategy_breakout.use_ma200_filter" in space.specs:
            genes["strategy_breakout.use_ma200_filter"] = bool(rng.random() < 0.2)
        if "strategy_breakout.min_rel_volume" in space.specs:
            genes["strategy_breakout.min_rel_volume"] = rng.uniform(1.0, 1.1)
        if "regime.adx_enter_threshold" in space.specs:
            genes["regime.adx_enter_threshold"] = rng.uniform(12.0, 20.0)
        if "regime.adx_trend_threshold" in space.specs:
            genes["regime.adx_trend_threshold"] = rng.uniform(18.0, 30.0)
        if "regime.adx_exit_threshold" in space.specs:
            genes["regime.adx_exit_threshold"] = rng.uniform(8.0, 16.0)
        if "strategy_breakout.breakout_lookback_N" in space.specs:
            genes["strategy_breakout.breakout_lookback_N"] = rng.randint(48, 160)
        if "strategy_breakout.atr_k" in space.specs:
            genes["strategy_breakout.atr_k"] = rng.uniform(1.8, 3.2)
    return _sanitize_individual(genes, space, rng)


def _save_checkpoint(path: Path, state: GAState, cache: dict[str, Any], cfg: GASettings) -> None:
    payload = {
        "generation": state.generation,
        "population": state.population,
        "hall_of_fame": state.hall_of_fame,
        "best_global": state.best_global,
        "last_generation_report": state.last_generation_report,
        "zero_trade_streak": state.zero_trade_streak,
        "cache": cache,
        "settings": asdict(cfg),
        "saved_at": int(time.time()),
    }
    write_json(path, payload)


def _load_checkpoint(path: Path) -> tuple[GAState, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    state = GAState(
        generation=int(payload.get("generation", 0)),
        population=list(payload.get("population", [])),
        hall_of_fame=list(payload.get("hall_of_fame", [])),
        best_global=payload.get("best_global"),
        last_generation_report=payload.get("last_generation_report"),
        zero_trade_streak=int(payload.get("zero_trade_streak", 0)),
    )
    cache = dict(payload.get("cache", {}))
    return state, cache


def _update_hof(hof: list[dict[str, Any]], results: list[EvalResult], limit: int = HOF_SIZE) -> list[dict[str, Any]]:
    by_hash: dict[str, dict[str, Any]] = {item["genes_hash"]: item for item in hof if "genes_hash" in item}
    for res in results:
        fitness_components = res.metrics.get("fitness_components", {}) if isinstance(res.metrics, dict) else {}
        candidate = {
            "genes_hash": res.genes_hash,
            "fitness": res.fitness,
            "genes": res.genes,
            "metrics": res.metrics,
            "run_dir": res.run_dir,
            "generation": res.generation,
            "index": res.index,
            "min_trades_hard": fitness_components.get("min_trades_hard"),
            "target_trades": fitness_components.get("target_trades"),
            "min_trades_for_sharpe": fitness_components.get("min_trades_for_sharpe"),
            "ret_term": fitness_components.get("ret_term"),
            "dd_term": fitness_components.get("dd_term"),
            "sharpe_term": fitness_components.get("sharpe_term"),
            "penalty_trades": fitness_components.get("penalty_trades"),
            "hard_cut_applied": fitness_components.get("hard_cut_applied"),
            "hard_cut_reason": fitness_components.get("hard_cut_reason"),
            "hard_filters_on": _count_hard_filters(res.genes),
            "filters": _hard_filter_bits(res.genes),
        }
        current = by_hash.get(res.genes_hash)
        if current is None or float(candidate["fitness"]) > float(current.get("fitness", float("-inf"))):
            by_hash[res.genes_hash] = candidate

    merged = sorted(by_hash.values(), key=lambda item: float(item.get("fitness", float("-inf"))), reverse=True)
    return merged[:limit]


def _persist_hof(path: Path, hof: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in hof]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _format_best_metrics(metrics: dict[str, Any]) -> str:
    summary = metrics.get("summary", {}) if isinstance(metrics, dict) else {}
    return (
        f"trades={int(summary.get('total_trades', 0))} "
        f"ret={float(summary.get('return_net', 0.0)):.4f} "
        f"dd={float(summary.get('max_drawdown', 0.0)):.4f} "
        f"sharpe={float(summary.get('sharpe', 0.0)):.3f} "
    )


def _format_fitness_breakdown(metrics: dict[str, Any]) -> str:
    parts = metrics.get("fitness_components", {}) if isinstance(metrics, dict) else {}
    hard_cut = bool(parts.get("hard_cut_applied", False))
    reason = str(parts.get("hard_cut_reason", "")) if hard_cut else ""
    return (
        f"ret_term={float(parts.get('ret_term', 0.0)):.4f} "
        f"dd_term={float(parts.get('dd_term', 0.0)):.4f} "
        f"sharpe_term={float(parts.get('sharpe_term', 0.0)):.4f} "
        f"penalty_trades={float(parts.get('penalty_trades', 0.0)):.4f} "
        f"hard_cut={'YES' if hard_cut else 'NO'}"
        + (f" reason={reason}" if reason else "")
    )


def _format_filters(genes: dict[str, Any]) -> str:
    bits = _hard_filter_bits(genes)
    hard_on = sum(bits.values())
    bits_str = ", ".join(f"{k}:{v}" for k, v in bits.items())
    return f"hard_filters_on={hard_on} filters={{" + bits_str + "}"


def _report_to_lines(report: dict[str, Any]) -> list[str]:
    line_sep = "[blue]" + "=" * 88 + "[/blue]"
    line_gen = (
        f"[cyan]GA gen[/cyan]={int(report.get('generation', -1))} evals={int(report.get('evals', 0))} "
        f"gen_time={float(report.get('gen_time', 0.0)):.2f}s best_gen={float(report.get('best_gen', float('-inf'))):.6f} "
        f"best_global={float(report.get('best_global', float('-inf'))):.6f}"
    )
    line_metrics = f"[cyan]Best gen metrics[/cyan] {str(report.get('best_metrics', ''))}"
    line_breakdown = f"[cyan]Fitness breakdown (gen)[/cyan] {str(report.get('best_breakdown', ''))}"
    line_global_metrics = f"[cyan]Best global metrics[/cyan] {str(report.get('best_global_metrics', ''))}"
    line_global_breakdown = f"[cyan]Fitness breakdown (global)[/cyan] {str(report.get('best_global_breakdown', ''))}"
    line_filters_gen = f"[cyan]Filters (gen)[/cyan] {str(report.get('best_filters', ''))}"
    line_filters_global = f"[cyan]Filters (global)[/cyan] {str(report.get('best_global_filters', ''))}"
    line_genes = f"[cyan]Best genes[/cyan] {json.dumps(report.get('best_genes', {}), ensure_ascii=False, sort_keys=True)}"
    line_run = f"[cyan]Best run[/cyan] {str(report.get('best_run', ''))}"
    return [
        line_sep,
        line_gen,
        line_metrics,
        line_breakdown,
        line_filters_gen,
        line_global_metrics,
        line_global_breakdown,
        line_filters_global,
        line_genes,
        line_run,
        line_sep,
    ]


def _evaluate_task(payload: dict[str, Any]) -> EvalResult:
    space = payload.get("space", _WORKER_CTX.get("space"))
    outdir = payload.get("outdir", _WORKER_CTX.get("outdir"))
    config_path = payload.get("config_path", _WORKER_CTX.get("config_path"))
    data_path = payload.get("data_path", _WORKER_CTX.get("data_path"))
    funding_path = payload.get("funding_path", _WORKER_CTX.get("funding_path"))
    objective = payload.get("objective", _WORKER_CTX.get("objective"))
    min_trades_hard = payload.get("min_trades_hard", _WORKER_CTX.get("min_trades_hard"))
    target_trades = payload.get("target_trades", _WORKER_CTX.get("target_trades"))
    min_trades_for_sharpe = payload.get("min_trades_for_sharpe", _WORKER_CTX.get("min_trades_for_sharpe"))
    lambda_trades = payload.get("lambda_trades", _WORKER_CTX.get("lambda_trades"))
    w_ret = payload.get("w_ret", _WORKER_CTX.get("w_ret"))
    w_dd = payload.get("w_dd", _WORKER_CTX.get("w_dd"))
    w_sharpe = payload.get("w_sharpe", _WORKER_CTX.get("w_sharpe"))
    eval_backend = payload.get("eval_backend", _WORKER_CTX.get("eval_backend", "inprocess"))
    save_full_artifacts = payload.get("save_full_artifacts", _WORKER_CTX.get("save_full_artifacts", False))
    return evaluate_candidate(
        generation=int(payload["generation"]),
        index=int(payload["index"]),
        genes=dict(payload["genes"]),
        space=space,
        outdir=Path(outdir),
        config_path=str(config_path),
        data_path=str(data_path),
        funding_path=funding_path,
        objective=str(objective),
        min_trades_hard=int(min_trades_hard),
        target_trades=int(target_trades),
        min_trades_for_sharpe=int(min_trades_for_sharpe),
        lambda_trades=float(lambda_trades),
        w_ret=float(w_ret),
        w_dd=float(w_dd),
        w_sharpe=float(w_sharpe),
        eval_backend=str(eval_backend),
        save_full_artifacts=bool(save_full_artifacts),
    )


def _init_worker(
    space: SearchSpace,
    outdir: str,
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
    eval_backend: str,
    save_full_artifacts: bool,
) -> None:
    _WORKER_CTX.clear()
    _WORKER_CTX.update(
        {
            "space": space,
            "outdir": outdir,
            "config_path": config_path,
            "data_path": data_path,
            "funding_path": funding_path,
            "objective": objective,
            "min_trades_hard": min_trades_hard,
            "target_trades": target_trades,
            "min_trades_for_sharpe": min_trades_for_sharpe,
            "lambda_trades": lambda_trades,
            "w_ret": w_ret,
            "w_dd": w_dd,
            "w_sharpe": w_sharpe,
            "eval_backend": eval_backend,
            "save_full_artifacts": save_full_artifacts,
        }
    )


def _evaluate_task_worker(task: dict[str, Any]) -> EvalResult:
    return evaluate_candidate(
        generation=int(task["generation"]),
        index=int(task["index"]),
        genes=dict(task["genes"]),
        space=_WORKER_CTX["space"],
        outdir=Path(_WORKER_CTX["outdir"]),
        config_path=str(_WORKER_CTX["config_path"]),
        data_path=str(_WORKER_CTX["data_path"]),
        funding_path=_WORKER_CTX.get("funding_path"),
        objective=str(_WORKER_CTX["objective"]),
        min_trades_hard=int(_WORKER_CTX["min_trades_hard"]),
        target_trades=int(_WORKER_CTX["target_trades"]),
        min_trades_for_sharpe=int(_WORKER_CTX["min_trades_for_sharpe"]),
        lambda_trades=float(_WORKER_CTX["lambda_trades"]),
        w_ret=float(_WORKER_CTX["w_ret"]),
        w_dd=float(_WORKER_CTX["w_dd"]),
        w_sharpe=float(_WORKER_CTX["w_sharpe"]),
        eval_backend=str(_WORKER_CTX.get("eval_backend", "inprocess")),
        save_full_artifacts=bool(_WORKER_CTX.get("save_full_artifacts", False)),
    )


def _evaluate_population(
    *,
    generation: int,
    population: list[dict[str, Any]],
    space: SearchSpace,
    cfg: GASettings,
    cache: dict[str, Any],
) -> list[EvalResult]:
    indexed_results: list[EvalResult | None] = [None] * len(population)
    tasks: list[dict[str, Any]] = []
    pending_by_hash: dict[str, int] = {}
    progress_total = len(population)
    progress_done = 0
    progress_started = time.perf_counter()
    spinner_frames = [".  ", ".. ", "...", ".. ", ".  ", "   "]
    spinner_idx = 0
    last_line = ""
    render_lock = threading.Lock()
    spinner_running = True

    def emit_progress(force: bool = False, allow_same: bool = False) -> None:
        nonlocal spinner_idx, last_line
        total = max(1, progress_total)
        pct = int((progress_done * 100) / total)
        elapsed = time.perf_counter() - progress_started
        frame = spinner_frames[spinner_idx % len(spinner_frames)]
        msg = f"GA gen={generation} progress={pct:3d}% ({progress_done}/{progress_total}) working{frame} elapsed={elapsed:.1f}s"
        with render_lock:
            if force and msg == last_line:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return
            if not force and not allow_same and msg == last_line:
                return
            last_line = msg
            sys.stdout.write("\r" + msg)
            if force:
                sys.stdout.write("\n")
            sys.stdout.flush()

    def _spinner_loop() -> None:
        nonlocal spinner_idx
        while spinner_running:
            spinner_idx += 1
            emit_progress(allow_same=True)
            time.sleep(1.0)

    spinner_thread = threading.Thread(target=_spinner_loop, daemon=True)
    spinner_thread.start()

    eval_limit = cfg.max_evals_per_gen if cfg.max_evals_per_gen is not None else len(population)
    emit_progress()

    for index, genes in enumerate(population):
        genes = _sanitize_individual(genes, space, rng=random.Random(generation * 100000 + index + 17))
        population[index] = genes
        if index >= eval_limit:
            indexed_results[index] = EvalResult(
                generation=generation,
                index=index,
                genes=genes,
                genes_hash=genes_hash(genes),
                fitness=float("-inf"),
                metrics={"skipped": "max_evals_per_gen"},
                run_dir="",
                error="skipped",
            )
            progress_done += 1
            emit_progress()
            continue

        ghash = genes_hash(genes)
        entry = cache.get(ghash)
        if entry is not None:
            indexed_results[index] = load_cache_entry(generation, index, genes, entry)
            progress_done += 1
            emit_progress()
            continue

        if ghash in pending_by_hash:
            source_index = pending_by_hash[ghash]
            indexed_results[index] = EvalResult(
                generation=generation,
                index=index,
                genes=genes,
                genes_hash=ghash,
                fitness=float("nan"),
                metrics={"duplicate_of": source_index},
                run_dir="",
                cached=True,
            )
            progress_done += 1
            emit_progress()
            continue

        tasks.append(
            {
                "generation": generation,
                "index": index,
                "genes": genes,
            }
        )
        pending_by_hash[ghash] = index

    if tasks:
        _init_worker(
            space,
            cfg.outdir,
            cfg.config_path,
            cfg.data_path,
            cfg.funding_path,
            cfg.fitness_objective,
            cfg.min_trades_hard,
            cfg.target_trades,
            cfg.min_trades_for_sharpe,
            cfg.lambda_trades,
            cfg.w_ret,
            cfg.w_dd,
            cfg.w_sharpe,
            cfg.eval_backend,
            cfg.save_full_artifacts,
        )
        # Avoid multiprocessing overhead when running with a single worker.
        use_pool = cfg.n_jobs > 1
        if use_pool:
            with mp.Pool(
                processes=cfg.n_jobs,
                initializer=_init_worker,
                initargs=(
                    space,
                    cfg.outdir,
                    cfg.config_path,
                    cfg.data_path,
                    cfg.funding_path,
                    cfg.fitness_objective,
                    cfg.min_trades_hard,
                    cfg.target_trades,
                    cfg.min_trades_for_sharpe,
                    cfg.lambda_trades,
                    cfg.w_ret,
                    cfg.w_dd,
                    cfg.w_sharpe,
                    cfg.eval_backend,
                    cfg.save_full_artifacts,
                ),
            ) as pool:
                computed: list[EvalResult] = []
                try:
                    for result in pool.imap_unordered(_evaluate_task_worker, tasks, chunksize=1):
                        computed.append(result)
                        progress_done += 1
                        emit_progress()
                except KeyboardInterrupt:
                    pool.terminate()
                    pool.join()
                    raise
        else:
            computed = []
            try:
                for task in tasks:
                    computed.append(_evaluate_task(task))
                    progress_done += 1
                    emit_progress()
            except KeyboardInterrupt:
                raise

        for result in computed:
            indexed_results[result.index] = result
            cache[result.genes_hash] = make_cache_entry(result)

        for index, item in enumerate(indexed_results):
            if item is None:
                continue
            duplicate_of = item.metrics.get("duplicate_of") if isinstance(item.metrics, dict) else None
            if duplicate_of is None:
                continue
            source = indexed_results[int(duplicate_of)]
            if source is None:
                continue
            indexed_results[index] = EvalResult(
                generation=generation,
                index=index,
                genes=population[index],
                genes_hash=source.genes_hash,
                fitness=source.fitness,
                metrics=source.metrics,
                run_dir=source.run_dir,
                cached=True,
                error=source.error,
            )

    spinner_running = False
    spinner_thread.join(timeout=1.0)
    progress_done = progress_total
    if f"progress=100%" in last_line:
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        emit_progress(force=True, allow_same=True)

    final = [item for item in indexed_results if item is not None]
    final.sort(key=lambda item: item.index)
    return final


def _next_population(
    *,
    ranked: list[EvalResult],
    space: SearchSpace,
    rng: random.Random,
    population_size: int,
    elite: int,
    tournament: int,
    cx_prob: float,
    mut_prob: float,
) -> list[dict[str, Any]]:
    next_pop: list[dict[str, Any]] = [dict(item.genes) for item in ranked[:elite]]
    keys = list(space.specs.keys())

    while len(next_pop) < population_size:
        p1 = _tournament_pick(rng, ranked, tournament)
        p2 = _tournament_pick(rng, ranked, tournament)

        if rng.random() < cx_prob:
            child = _uniform_crossover(rng, p1, p2, keys)
        else:
            child = dict(p1)

        child = _mutate_individual(rng, child, space, mut_prob)
        next_pop.append(_sanitize_individual(child, space, rng))

    return next_pop[:population_size]


def _build_initial_population(space: SearchSpace, cfg: GASettings, rng: random.Random) -> list[dict[str, Any]]:
    population: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_candidate(genes: dict[str, Any]) -> None:
        normalized = _sanitize_individual(genes, space, rng)
        ghash = genes_hash(normalized)
        if ghash in seen:
            return
        seen.add(ghash)
        population.append(normalized)

    base = space.base_genes()
    baseline_target = int(round(cfg.population * max(0.0, min(1.0, cfg.init_baseline_ratio))))
    attempts = 0
    while len(population) < baseline_target and attempts < cfg.population * 20:
        add_candidate(_make_seed_individual(base, space, rng, mode=cfg.init_baseline_seed_mode))
        attempts += 1

    while len(population) < cfg.population:
        add_candidate(_sanitize_individual(space.sample_individual(rng), space, rng))
    return population[: cfg.population]


def _extract_total_trades(result: EvalResult) -> int:
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    summary = metrics.get("summary", {}) if isinstance(metrics, dict) else {}
    raw = summary.get("total_trades", summary.get("trades", 0))
    try:
        return int(raw)
    except Exception:
        return 0


def _build_zero_trade_escape_population(space: SearchSpace, cfg: GASettings, rng: random.Random) -> list[dict[str, Any]]:
    population: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_candidate(genes: dict[str, Any]) -> None:
        normalized = _sanitize_individual(genes, space, rng)
        ghash = genes_hash(normalized)
        if ghash in seen:
            return
        seen.add(ghash)
        population.append(normalized)

    base = space.base_genes()
    if base:
        add_candidate(base)

    profiles: list[dict[str, Any]] = [
        {
            "strategy_breakout.mode": "breakout",
            "strategy_breakout.breakout_lookback_N": 18,
            "strategy_breakout.atr_k": 1.2,
            "strategy_breakout.use_ma200_filter": False,
            "strategy_breakout.use_rel_volume_filter": False,
            "strategy_breakout.trade_direction": "both",
            "strategy_router.enable_range": True,
            "strategy_router.enable_chaos": True,
            "strategy_router.cooldown_bars_after_exit": 0,
            "regime.adx_trend_threshold": 14,
            "regime.adx_enter_threshold": 14,
            "regime.adx_exit_threshold": 10,
            "router.enabled": False,
            "market_structure.enabled": False,
            "funding_filter.enabled": False,
            "multi_timeframe.enabled": False,
            "time_exit.enabled": True,
            "time_exit.max_holding_hours": 18,
        },
        {
            "strategy_breakout.mode": "breakout",
            "strategy_breakout.breakout_lookback_N": 24,
            "strategy_breakout.atr_k": 1.6,
            "strategy_breakout.use_ma200_filter": False,
            "strategy_breakout.use_rel_volume_filter": False,
            "strategy_breakout.trade_direction": "short",
            "regime.adx_trend_threshold": 16,
            "regime.adx_enter_threshold": 16,
            "regime.adx_exit_threshold": 12,
            "router.enabled": False,
            "market_structure.enabled": False,
            "funding_filter.enabled": False,
        },
        {
            "strategy_breakout.mode": "breakout",
            "strategy_breakout.breakout_lookback_N": 30,
            "strategy_breakout.atr_k": 1.8,
            "strategy_breakout.use_ma200_filter": False,
            "strategy_breakout.use_rel_volume_filter": False,
            "strategy_breakout.trade_direction": "long",
            "regime.adx_trend_threshold": 16,
            "regime.adx_enter_threshold": 16,
            "regime.adx_exit_threshold": 12,
            "router.enabled": False,
            "market_structure.enabled": False,
            "funding_filter.enabled": False,
        },
    ]
    for profile in profiles:
        seeded = dict(base)
        for key, value in profile.items():
            if key in space.specs:
                seeded[key] = value
        add_candidate(seeded)

    while len(population) < cfg.population:
        candidate = space.sample_individual(rng)
        if "strategy_breakout.use_ma200_filter" in candidate and rng.random() < 0.7:
            candidate["strategy_breakout.use_ma200_filter"] = False
        if "strategy_breakout.use_rel_volume_filter" in candidate and rng.random() < 0.7:
            candidate["strategy_breakout.use_rel_volume_filter"] = False
        if "market_structure.enabled" in candidate and rng.random() < 0.7:
            candidate["market_structure.enabled"] = False
        if "funding_filter.enabled" in candidate and rng.random() < 0.8:
            candidate["funding_filter.enabled"] = False
        if "router.enabled" in candidate and rng.random() < 0.7:
            candidate["router.enabled"] = False
        if "strategy_breakout.breakout_lookback_N" in candidate:
            candidate["strategy_breakout.breakout_lookback_N"] = min(
                int(candidate["strategy_breakout.breakout_lookback_N"]),
                36,
            )
        add_candidate(candidate)
    return population[: cfg.population]


def run_genetic_optimization(cfg: GASettings) -> None:
    os.environ.setdefault("BOT_LOG_LEVEL", "ERROR")
    logging.getLogger("bot").setLevel(logging.WARNING)
    logging.getLogger("bot.strategy").setLevel(logging.ERROR)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = outdir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / "ga_state.json"
    cache_path = outdir / "cache_index.json"
    hof_path = outdir / "hof.jsonl"
    best_path = outdir / "best.yaml"

    base_config = _load_yaml(Path(cfg.config_path))
    space = discover_search_space(base_config, ga_space_path=cfg.ga_space_path)

    rng = random.Random(cfg.seed)
    latest_gen = _detect_latest_generation(outdir)
    start_from_dirs = (latest_gen + 1) if latest_gen is not None else 0

    if checkpoint_path.exists():
        state, cache = _load_checkpoint(checkpoint_path)
        state.generation = max(state.generation, start_from_dirs)
        if state.population:
            print(
                f"[yellow]GA resume[/yellow] generation={state.generation} "
                f"population={len(state.population)} source=checkpoint"
            )
        else:
            state = GAState(generation=start_from_dirs, population=[], hall_of_fame=[], best_global=None)
    else:
        cache = {}
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                cache = json.load(f)
        state = GAState(generation=start_from_dirs, population=[], hall_of_fame=[], best_global=None)
        if start_from_dirs > 0:
            print(f"[yellow]GA resume[/yellow] generation={start_from_dirs} source=existing_gen_dirs")

    if not state.population:
        state.population = _build_initial_population(space, cfg, rng)

    generation = state.generation
    last_generation_lines: list[str] | None = None
    if state.last_generation_report:
        last_generation_lines = _report_to_lines(state.last_generation_report)
        for line in last_generation_lines:
            print(line)
    try:
        while True:
            if cfg.max_generations is not None and generation >= cfg.max_generations:
                break

            _clear_console()
            if last_generation_lines:
                for line in last_generation_lines:
                    print(line)
            gen_start = time.perf_counter()
            results = _evaluate_population(
                generation=generation,
                population=state.population,
                space=space,
                cfg=cfg,
                cache=cache,
            )
            results.sort(key=lambda item: item.fitness, reverse=True)

            state.hall_of_fame = _update_hof(state.hall_of_fame, results)
            best_gen = results[0]
            best_gen_trades = _extract_total_trades(best_gen)
            if best_gen_trades <= 0:
                state.zero_trade_streak += 1
            else:
                state.zero_trade_streak = 0
            if state.best_global is None or float(best_gen.fitness) > float(state.best_global.get("fitness", float("-inf"))):
                state.best_global = {
                    "generation": generation,
                    "index": best_gen.index,
                    "genes_hash": best_gen.genes_hash,
                    "fitness": best_gen.fitness,
                    "genes": best_gen.genes,
                    "metrics": best_gen.metrics,
                    "run_dir": best_gen.run_dir,
                }

            if generation % max(1, cfg.save_best_every) == 0 and state.best_global is not None:
                best_cfg = space.apply_genes(dict(state.best_global["genes"]))
                write_yaml(best_path, best_cfg)

            _persist_hof(hof_path, state.hall_of_fame)
            write_json(cache_path, cache)

            gen_time = time.perf_counter() - gen_start
            if generation % max(1, cfg.print_every) == 0:
                global_best = state.best_global or {}
                global_metrics = global_best.get("metrics", {}) if isinstance(global_best, dict) else {}
                report = {
                    "generation": generation,
                    "evals": len(results),
                    "gen_time": gen_time,
                    "best_gen": best_gen.fitness,
                    "best_global": float(global_best.get("fitness", float("-inf"))),
                    "best_metrics": _format_best_metrics(best_gen.metrics),
                    "best_breakdown": _format_fitness_breakdown(best_gen.metrics),
                    "best_filters": _format_filters(best_gen.genes),
                    "best_global_metrics": _format_best_metrics(global_metrics),
                    "best_global_breakdown": _format_fitness_breakdown(global_metrics),
                    "best_global_filters": _format_filters(global_best.get("genes", {}) if isinstance(global_best, dict) else {}),
                    "best_genes": best_gen.genes,
                    "best_run": best_gen.run_dir,
                }
                state.last_generation_report = report
                lines = _report_to_lines(report)
                for line in lines:
                    print(line)
                last_generation_lines = lines

            next_population = _next_population(
                ranked=results,
                space=space,
                rng=rng,
                population_size=cfg.population,
                elite=max(1, min(cfg.elite, cfg.population)),
                tournament=max(2, min(cfg.tournament, cfg.population)),
                cx_prob=cfg.cx_prob,
                mut_prob=cfg.mut_prob,
            )

            if state.zero_trade_streak >= 2:
                print(
                    "[yellow]GA rescue[/yellow] zero-trades por "
                    f"{state.zero_trade_streak} gerações: aplicando escape population."
                )
                next_population = _build_zero_trade_escape_population(space, cfg, rng)
                state.zero_trade_streak = 0

            generation += 1
            state.generation = generation
            state.population = next_population

            _save_checkpoint(checkpoint_path, state, cache, cfg)
    except KeyboardInterrupt:
        print("\n[yellow]Ctrl+C detectado. Salvando checkpoint e encerrando...[/yellow]")
    finally:
        _save_checkpoint(checkpoint_path, state, cache, cfg)
        write_json(cache_path, cache)
        _persist_hof(hof_path, state.hall_of_fame)
