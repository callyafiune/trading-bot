from pathlib import Path

from bot.ga.ga import _detect_latest_generation


def test_detect_latest_generation(tmp_path: Path) -> None:
    (tmp_path / "gen_00000").mkdir()
    (tmp_path / "gen_00012").mkdir()
    (tmp_path / "x").mkdir()
    assert _detect_latest_generation(tmp_path) == 12
