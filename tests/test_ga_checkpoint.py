from pathlib import Path

from bot.ga.ga import GASettings, GAState, _load_checkpoint, _save_checkpoint


def test_checkpoint_payload_shape(tmp_path: Path) -> None:
    cfg = GASettings(
        data_path="data.parquet",
        funding_path=None,
        config_path="config/settings.yaml",
        outdir=str(tmp_path),
        population=4,
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
        min_trades=12,
        ga_space_path=None,
        eval_backend="inprocess",
    )

    checkpoint = tmp_path / "ga_state.json"
    state = GAState(
        generation=3,
        population=[{"a": 1}],
        hall_of_fame=[{"genes_hash": "h", "fitness": 1.0}],
        best_global={"genes_hash": "h", "fitness": 1.0},
    )
    cache = {"h": {"fitness": 1.0}}

    _save_checkpoint(checkpoint, state, cache, cfg)
    loaded_state, loaded_cache = _load_checkpoint(checkpoint)

    assert loaded_state.generation == 3
    assert loaded_state.population[0]["a"] == 1
    assert loaded_state.best_global["genes_hash"] == "h"
    assert loaded_cache["h"]["fitness"] == 1.0
