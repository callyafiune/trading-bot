from typer.testing import CliRunner

from bot.cli import app


def test_compare_requires_two_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["compare", "--runs", "runs/only_one"])
    assert result.exit_code != 0
    assert "Informe ao menos duas runs" in result.output
