from bot.ga.eval import EvalResult, compute_fitness
from bot.ga.ga import _update_hof


def test_compute_fitness_hard_cut_for_low_trades() -> None:
    score, components, _ = compute_fitness(
        {
            "return_net": 0.0061,
            "max_drawdown": -0.0020,
            "sharpe": 1.79,
            "total_trades": 7,
        },
        objective="score",
        min_trades_hard=30,
        target_trades=140,
        min_trades_for_sharpe=120,
        lambda_trades=6.0,
        w_ret=1.0,
        w_dd=0.6,
        w_sharpe=10.0,
    )
    assert score == -10_000.0
    assert components["hard_cut_applied"] is True
    assert components["hard_cut_reason"] == "invalid_low_trades_hard"
    assert components["invalid_low_trades_hard"] is True
    assert components["penalty_trades"] > 0.0


def test_compute_fitness_uses_sharpe_with_enough_trades() -> None:
    score, components, _ = compute_fitness(
        {
            "return_net": 0.02,
            "max_drawdown": -0.03,
            "sharpe": 1.5,
            "total_trades": 200,
        },
        objective="score",
        min_trades_hard=30,
        target_trades=140,
        min_trades_for_sharpe=120,
        lambda_trades=6.0,
        w_ret=1.0,
        w_dd=0.6,
        w_sharpe=10.0,
    )
    expected = (1.0 * 2.0) - (0.6 * 3.0) + (10.0 * 1.5) - 0.0
    assert abs(score - expected) < 1e-9
    assert abs(components["sharpe_term"] - 15.0) < 1e-9
    assert components["hard_cut_applied"] is False


def test_compute_fitness_disables_sharpe_for_small_sample() -> None:
    score, components, _ = compute_fitness(
        {
            "return_net": 0.02,
            "max_drawdown": -0.03,
            "sharpe": 5.0,
            "total_trades": 90,
        },
        objective="score",
        min_trades_hard=30,
        target_trades=140,
        min_trades_for_sharpe=100,
        lambda_trades=6.0,
        w_ret=1.0,
        w_dd=0.6,
        w_sharpe=10.0,
    )
    penalty = 6.0 * ((140.0 - 90.0) / 140.0) ** 2
    expected = (1.0 * 2.0) - (0.6 * 3.0) - penalty
    assert abs(score - expected) < 1e-9
    assert components["sharpe_term"] == 0.0


def test_compute_fitness_progressive_trade_penalty_without_hard_cut() -> None:
    score, components, _ = compute_fitness(
        {
            "return_net": 0.01,
            "max_drawdown": -0.02,
            "sharpe": 1.0,
            "total_trades": 40,
        },
        objective="score",
        min_trades_hard=30,
        target_trades=140,
        min_trades_for_sharpe=120,
        lambda_trades=6.0,
        w_ret=1.0,
        w_dd=0.6,
        w_sharpe=10.0,
    )
    assert score > -10_000.0
    assert components["hard_cut_applied"] is False
    assert components["penalty_trades"] > 0.0


def test_compute_fitness_penalty_zero_at_target_trades() -> None:
    _, components, _ = compute_fitness(
        {
            "return_net": 0.01,
            "max_drawdown": -0.02,
            "sharpe": 1.0,
            "total_trades": 140,
        },
        objective="score",
        min_trades_hard=30,
        target_trades=140,
        min_trades_for_sharpe=120,
        lambda_trades=6.0,
        w_ret=1.0,
        w_dd=0.6,
        w_sharpe=10.0,
    )
    assert components["penalty_trades"] == 0.0


def test_hof_serialization_includes_fitness_breakdown_fields() -> None:
    result = EvalResult(
        generation=1,
        index=0,
        genes={"a": 1},
        genes_hash="abc",
        fitness=1.23,
        metrics={
            "fitness_components": {
                "min_trades_hard": 30,
                "target_trades": 140,
                "min_trades_for_sharpe": 80,
                "ret_term": 2.0,
                "dd_term": 1.0,
                "sharpe_term": 0.0,
                "penalty_trades": 0.4,
                "hard_cut_applied": False,
                "hard_cut_reason": "",
            }
        },
        run_dir="runs/x",
    )
    hof = _update_hof([], [result], limit=5)
    assert len(hof) == 1
    row = hof[0]
    assert row["min_trades_hard"] == 30
    assert row["target_trades"] == 140
    assert row["min_trades_for_sharpe"] == 80
    assert row["ret_term"] == 2.0
    assert row["dd_term"] == 1.0
    assert row["sharpe_term"] == 0.0
    assert row["penalty_trades"] == 0.4
    assert row["hard_cut_applied"] is False
