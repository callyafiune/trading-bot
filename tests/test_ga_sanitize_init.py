import random

from bot.ga.ga import GASettings, _build_initial_population, _count_hard_filters, _sanitize_individual
from bot.ga.space import discover_search_space


def _build_test_space(tmp_path):
    base = {
        "funding_filter": {"enabled": False},
        "market_structure": {
            "enabled": False,
            "left_bars": 3,
            "right_bars": 3,
            "gate": {
                "enabled": False,
                "mode": "msb_only",
                "block_in_neutral": False,
                "hybrid_require_both": False,
            },
            "msb": {"enabled": False},
        },
        "multi_timeframe": {"enabled": False},
        "router": {"enabled": False},
        "regime": {
            "adx_exit_threshold": 10.0,
            "adx_enter_threshold": 16.0,
            "adx_trend_threshold": 24.0,
        },
        "strategy_breakout": {
            "use_ma200_filter": False,
            "use_rel_volume_filter": False,
            "min_rel_volume": 1.05,
            "breakout_lookback_N": 72,
            "atr_k": 2.5,
        },
    }
    space_yaml = tmp_path / "ga_space.yaml"
    space_yaml.write_text(
        "parameters:\n"
        "  funding_filter.enabled:\n"
        "    type: bool\n"
        "  market_structure.enabled:\n"
        "    type: bool\n"
        "  market_structure.gate.enabled:\n"
        "    type: bool\n"
        "  market_structure.gate.mode:\n"
        "    type: choice\n"
        "    choices: [msb_only, structure_trend, hybrid]\n"
        "  market_structure.gate.block_in_neutral:\n"
        "    type: bool\n"
        "  market_structure.gate.hybrid_require_both:\n"
        "    type: bool\n"
        "  market_structure.left_bars:\n"
        "    type: int\n"
        "    min: 2\n"
        "    max: 8\n"
        "  market_structure.right_bars:\n"
        "    type: int\n"
        "    min: 2\n"
        "    max: 8\n"
        "  market_structure.msb.enabled:\n"
        "    type: bool\n"
        "  multi_timeframe.enabled:\n"
        "    type: bool\n"
        "  router.enabled:\n"
        "    type: bool\n"
        "  regime.adx_exit_threshold:\n"
        "    type: float\n"
        "    min: 8\n"
        "    max: 20\n"
        "  regime.adx_enter_threshold:\n"
        "    type: float\n"
        "    min: 12\n"
        "    max: 24\n"
        "  regime.adx_trend_threshold:\n"
        "    type: float\n"
        "    min: 18\n"
        "    max: 35\n"
        "  strategy_breakout.use_ma200_filter:\n"
        "    type: bool\n"
        "  strategy_breakout.use_rel_volume_filter:\n"
        "    type: bool\n"
        "  strategy_breakout.min_rel_volume:\n"
        "    type: float\n"
        "    min: 1.0\n"
        "    max: 1.4\n"
        "  strategy_breakout.breakout_lookback_N:\n"
        "    type: int\n"
        "    min: 48\n"
        "    max: 240\n"
        "  strategy_breakout.atr_k:\n"
        "    type: float\n"
        "    min: 1.5\n"
        "    max: 4.0\n",
        encoding="utf-8",
    )
    return discover_search_space(base, ga_space_path=space_yaml)


def test_sanitize_repairs_adx_and_filter_pile(tmp_path) -> None:
    space = _build_test_space(tmp_path)
    genes = {
        "funding_filter.enabled": True,
        "market_structure.enabled": True,
        "market_structure.gate.enabled": True,
        "market_structure.gate.mode": "hybrid",
        "market_structure.gate.block_in_neutral": True,
        "market_structure.gate.hybrid_require_both": True,
        "market_structure.left_bars": 99,
        "market_structure.right_bars": -4,
        "market_structure.msb.enabled": True,
        "multi_timeframe.enabled": True,
        "router.enabled": True,
        "regime.adx_exit_threshold": 20.0,
        "regime.adx_enter_threshold": 12.0,
        "regime.adx_trend_threshold": 18.0,
        "strategy_breakout.use_ma200_filter": True,
        "strategy_breakout.use_rel_volume_filter": True,
        "strategy_breakout.min_rel_volume": 1.4,
        "strategy_breakout.breakout_lookback_N": 80,
        "strategy_breakout.atr_k": 2.4,
    }
    sanitized = _sanitize_individual(genes, space, random.Random(42))

    assert sanitized["regime.adx_exit_threshold"] <= sanitized["regime.adx_enter_threshold"] <= sanitized[
        "regime.adx_trend_threshold"
    ]
    assert 2 <= sanitized["market_structure.left_bars"] <= 8
    assert 2 <= sanitized["market_structure.right_bars"] <= 8
    assert sanitized["market_structure.gate.hybrid_require_both"] is False
    assert sanitized["market_structure.gate.block_in_neutral"] is False
    assert _count_hard_filters(sanitized) <= 2


def test_sanitize_disables_gate_and_msb_when_market_structure_off(tmp_path) -> None:
    space = _build_test_space(tmp_path)
    genes = {
        "market_structure.enabled": False,
        "market_structure.gate.enabled": True,
        "market_structure.msb.enabled": True,
    }
    sanitized = _sanitize_individual(genes, space, random.Random(7))
    assert sanitized["market_structure.enabled"] is False
    assert sanitized["market_structure.gate.enabled"] is False
    assert sanitized["market_structure.msb.enabled"] is False


def test_initial_population_has_baseline_ratio(tmp_path) -> None:
    space = _build_test_space(tmp_path)
    cfg = GASettings(
        data_path="data.parquet",
        funding_path=None,
        config_path="config/settings.yaml",
        outdir=str(tmp_path),
        population=10,
        elite=1,
        tournament=2,
        cx_prob=0.4,
        mut_prob=0.6,
        seed=42,
        n_jobs=1,
        resume=False,
        fitness_objective="score",
        max_generations=1,
        max_evals_per_gen=None,
        print_every=1,
        save_best_every=1,
        min_trades_hard=30,
        target_trades=140,
        init_baseline_ratio=0.7,
        init_baseline_seed_mode="baseline",
    )

    population = _build_initial_population(space, cfg, random.Random(123))
    baseline_count = sum(
        1
        for genes in population
        if not bool(genes.get("funding_filter.enabled", False))
        and not bool(genes.get("market_structure.enabled", False))
        and not bool(genes.get("market_structure.gate.enabled", False))
        and not bool(genes.get("market_structure.msb.enabled", False))
        and not bool(genes.get("strategy_breakout.use_rel_volume_filter", False))
        and not bool(genes.get("router.enabled", False))
    )
    assert len(population) == 10
    assert baseline_count >= 7
