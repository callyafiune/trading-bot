from bot.ga.eval import load_cache_entry, make_cache_entry
from bot.ga.space import genes_hash


def test_genes_hash_is_stable_for_key_order() -> None:
    g1 = {"a": 1, "b": True, "c": 0.5}
    g2 = {"c": 0.5, "b": True, "a": 1}
    assert genes_hash(g1) == genes_hash(g2)


def test_cache_roundtrip_entry() -> None:
    result = load_cache_entry(
        generation=1,
        index=2,
        genes={"x": 1},
        cache_entry={"genes_hash": "abc", "fitness": 2.5, "metrics": {"m": 1}, "run_dir": "runs/x"},
    )
    cached = make_cache_entry(result)
    assert cached["genes_hash"] == "abc"
    assert cached["fitness"] == 2.5
    assert cached["run_dir"] == "runs/x"
