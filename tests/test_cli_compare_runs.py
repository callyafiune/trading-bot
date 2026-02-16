from bot.cli import _collect_compare_runs


def test_collect_compare_runs_accepts_option_and_extra_args() -> None:
    runs = _collect_compare_runs(["runs/a"], ["runs/b"])
    assert runs == ["runs/a", "runs/b"]


def test_collect_compare_runs_accepts_comma_separated_values() -> None:
    runs = _collect_compare_runs(["runs/a,runs/b"], [])
    assert runs == ["runs/a", "runs/b"]
