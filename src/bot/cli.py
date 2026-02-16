from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import typer
import yaml
from rich import print
from rich.console import Console
from rich.table import Table

from bot.backtest.engine import BacktestEngine
from bot.backtest.metrics import compute_metrics
from bot.backtest.reporting import print_summary
from bot.execution.broker_binance import BrokerBinance
from bot.features.builder import build_features
from bot.market_data.binance_client import BinanceDataClient
from bot.market_data.loader import load_parquet, save_parquet
from bot.regime.detector import RegimeDetector
from bot.runs.run_manager import RunManager
from bot.runs.serializers import write_json
from bot.utils.config import Settings, load_settings
from bot.utils.logging import setup_logger

app = typer.Typer()
logger = setup_logger()


def _normalize_param_key(key: str) -> str:
    if key.startswith("strategy."):
        return key.replace("strategy.", "strategy_breakout.", 1)
    if key.startswith("breakout."):
        return key.replace("breakout.", "strategy_breakout.", 1)
    return key


def _set_nested_value(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = _normalize_param_key(dotted_key).split(".")
    cursor = payload
    for k in keys[:-1]:
        if k not in cursor or not isinstance(cursor[k], dict):
            cursor[k] = {}
        cursor = cursor[k]
    cursor[keys[-1]] = value


def _parse_param_option(option: str) -> tuple[str, list[Any]]:
    if "=" not in option:
        raise typer.BadParameter(f"Formato inválido em --param: {option}")
    key, values_str = option.split("=", 1)
    values = [yaml.safe_load(v) for v in values_str.split(",") if v != ""]
    if not values:
        raise typer.BadParameter(f"Nenhum valor fornecido em --param: {option}")
    return key, values


def _collect_compare_runs(primary: list[str], extra: list[str]) -> list[str]:
    runs: list[str] = []
    for item in [*primary, *extra]:
        if not item:
            continue
        runs.extend([token.strip() for token in item.split(",") if token.strip()])
    return runs


def _build_summary(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    summary = dict(metrics)
    summary["total_trades"] = int(len(trades))
    summary["start_ts"] = str(equity["timestamp"].min()) if not equity.empty else None
    summary["end_ts"] = str(equity["timestamp"].max()) if not equity.empty else None
    summary["fees_total"] = float(trades["fees"].sum()) if not trades.empty else 0.0
    summary["slippage_total"] = float(trades["slippage"].sum()) if not trades.empty else 0.0
    summary["interest_total"] = float(trades["interest"].sum()) if not trades.empty else 0.0
    summary["long_trades"] = int((trades["direction"] == "LONG").sum()) if not trades.empty else 0
    summary["short_trades"] = int((trades["direction"] == "SHORT").sum()) if not trades.empty else 0
    summary["pnl_long"] = float(trades.loc[trades["direction"] == "LONG", "pnl_net"].sum()) if not trades.empty else 0.0
    summary["pnl_short"] = float(trades.loc[trades["direction"] == "SHORT", "pnl_net"].sum()) if not trades.empty else 0.0
    summary["counts"] = {
        "signals_total": int(diagnostics.get("signals_total", 0)),
        "entries_executed": int(diagnostics.get("entries_executed", 0)),
        "blocked_regime": int(diagnostics.get("signals_blocked_regime", 0)),
        "blocked_risk": int(diagnostics.get("signals_blocked_risk", 0)),
        "blocked_killswitch": int(diagnostics.get("signals_blocked_killswitch", 0)),
        "blocked_mode": int(diagnostics.get("signals_blocked_mode", 0)),
        "blocked_ma200": int(diagnostics.get("signals_blocked_ma200", 0)),
        "killswitch_events": int(diagnostics.get("killswitch_events", 0)),
    }
    return summary


def _build_regime_stats(df: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    regimes = ["TREND", "RANGE", "CHAOS"]
    total_candles = len(df)
    result: dict[str, Any] = {}

    for regime in regimes:
        candles_count = int((df["regime"] == regime).sum()) if "regime" in df.columns else 0
        regime_trades = trades.loc[trades["regime_at_entry"] == regime] if not trades.empty else pd.DataFrame()
        trades_count = int(len(regime_trades))
        pnl_total = float(regime_trades["pnl_net"].sum()) if trades_count else 0.0
        win_rate = float((regime_trades["pnl_net"] > 0).mean()) if trades_count else 0.0
        avg_pnl = float(regime_trades["pnl_net"].mean()) if trades_count else 0.0
        result[regime] = {
            "candles_count": candles_count,
            "pct_time": float(candles_count / total_candles) if total_candles else 0.0,
            "trades_count": trades_count,
            "pnl_net_total": pnl_total,
            "avg_pnl": avg_pnl,
            "win_rate": win_rate,
        }
    return result


def _build_direction_stats(trades: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for direction in ["LONG", "SHORT"]:
        direction_trades = trades.loc[trades["direction"] == direction] if not trades.empty else pd.DataFrame()
        trades_count = int(len(direction_trades))
        result[direction] = {
            "trades_count": trades_count,
            "pnl_net_total": float(direction_trades["pnl_net"].sum()) if trades_count else 0.0,
            "avg_pnl": float(direction_trades["pnl_net"].mean()) if trades_count else 0.0,
            "win_rate": float((direction_trades["pnl_net"] > 0).mean()) if trades_count else 0.0,
            "avg_hold_hours": float(direction_trades["hold_hours"].mean()) if trades_count else 0.0,
        }
    return result


def _execute_backtest(
    data_path: str,
    cfg: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    df = load_parquet(data_path)
    if df.empty:
        print("[red]Dataset vazio. Verifique o arquivo de entrada.[/red]")
        raise typer.Exit(code=1)

    print(
        "[cyan]Dataset:[/cyan]"
        f" candles={len(df)}"
        f" | inicio={df['open_time'].min()}"
        f" | fim={df['open_time'].max()}"
    )

    df = build_features(df)
    df["regime"] = RegimeDetector(cfg.regime).apply(df)
    trend_candles = int((df["regime"] == "TREND").sum())
    trend_pct = (trend_candles / len(df)) * 100 if len(df) else 0.0
    print(f"[cyan]Regime:[/cyan] TREND={trend_candles} ({trend_pct:.2f}%)")

    engine = BacktestEngine(cfg)
    trades, equity = engine.run(df)
    print(f"[cyan]Backtest:[/cyan] total_trades={len(trades)}")
    if engine.last_run_diagnostics:
        diag = engine.last_run_diagnostics
        print(
            "[cyan]Diagnostics:[/cyan]"
            f" signals_total={diag.get('signals_total', 0)}"
            f" | blocked_regime={diag.get('signals_blocked_regime', 0)}"
            f" | blocked_risk={diag.get('signals_blocked_risk', 0)}"
            f" | blocked_killswitch={diag.get('signals_blocked_killswitch', 0)}"
            f" | blocked_mode={diag.get('signals_blocked_mode', 0)}"
            f" | blocked_ma200={diag.get('signals_blocked_ma200', 0)}"
            f" | entries_executed={diag.get('entries_executed', 0)}"
            f" | killswitch_events={diag.get('killswitch_events', 0)}"
            f" | first_killswitch_at={diag.get('first_killswitch_at')}"
        )

    metrics = compute_metrics(trades, equity)
    summary = _build_summary(trades, equity, metrics, engine.last_run_diagnostics)
    regime_stats = _build_regime_stats(df, trades)
    direction_stats = _build_direction_stats(trades)
    return trades, equity, summary, regime_stats, direction_stats


@app.command("fetch-data")
def fetch_data(start: str = typer.Option(None), end: str = typer.Option(None), config: str = "config/settings.yaml"):
    cfg = load_settings(config)
    start_date = start or cfg.start_date
    end_date = end or cfg.end_date
    df = BinanceDataClient().fetch_ohlcv(cfg.symbol, cfg.interval, start_date, end_date)
    raw_path = Path(f"data/raw/{cfg.symbol}_{cfg.interval}_{start_date}_{end_date}.parquet")
    proc_path = Path(f"data/processed/{cfg.symbol}_{cfg.interval}_{start_date}_{end_date}.parquet")
    save_parquet(df, raw_path, proc_path, cfg.interval)
    print(f"Dados salvos em {raw_path} e {proc_path}")


@app.command("backtest")
def backtest(
    data_path: str = typer.Option(..., "--data-path"),
    config: str = typer.Option("config/settings.yaml", "--config"),
    outdir: str = typer.Option("runs", "--outdir"),
    run_name: str | None = typer.Option(None, "--run-name"),
    seed: int | None = typer.Option(None, "--seed"),
    tag: str | None = typer.Option(None, "--tag"),
    mode: str | None = typer.Option(None, "--mode"),
    atr_k: float | None = typer.Option(None, "--atr-k"),
    breakout_N: int | None = typer.Option(None, "--breakout-N"),
    adx_threshold: float | None = typer.Option(None, "--adx-threshold"),
    use_ma200_filter: bool | None = typer.Option(None, "--use-ma200-filter/--no-use-ma200-filter"),
    ml_threshold: float | None = typer.Option(None, "--ml-threshold"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    cfg = load_settings(config)
    if mode is not None:
        cfg.strategy_breakout.mode = mode  # type: ignore[assignment]
    if atr_k is not None:
        cfg.strategy_breakout.atr_k = atr_k
    if breakout_N is not None:
        cfg.strategy_breakout.breakout_lookback_N = breakout_N
    if adx_threshold is not None:
        cfg.regime.adx_trend_threshold = adx_threshold
    if use_ma200_filter is not None:
        cfg.strategy_breakout.use_ma200_filter = use_ma200_filter
    if ml_threshold is not None:
        cfg.strategy_breakout.ml_prob_threshold = ml_threshold
    manager = RunManager(outdir)
    ctx = manager.build_context(cfg, run_name=run_name)

    if dry_run:
        print(f"[yellow]DRY RUN[/yellow] run_dir={ctx.run_dir} params_hash={ctx.params_hash}")
        return

    manager.init_run(ctx, settings=cfg, data_path=data_path, config_path=config, seed=seed, tag=tag)
    trades, equity, summary, regime_stats, direction_stats = _execute_backtest(data_path, cfg)
    manager.persist_outputs(
        ctx,
        summary=summary,
        trades=trades,
        equity=equity,
        regime_stats=regime_stats,
        direction_stats=direction_stats,
    )
    print_summary(summary)
    print(f"[green]Artefatos salvos em:[/green] {ctx.run_dir}")


@app.command("compare")
def compare(
    runs: list[str] = typer.Option([], "--runs", help="Runs para comparar (repetível ou separado por vírgula)."),
    extra_runs: list[str] = typer.Argument([], help="Runs adicionais aceitos após --runs para compatibilidade."),
    save_path: str | None = typer.Option(None, "--save-path"),
):
    resolved_runs = _collect_compare_runs(runs, extra_runs)
    if len(resolved_runs) < 2:
        raise typer.BadParameter("Informe ao menos duas runs em --runs")

    summaries: list[tuple[str, dict[str, Any]]] = []
    for run in resolved_runs:
        summary_path = Path(run) / "summary.json"
        if not summary_path.exists():
            raise typer.BadParameter(f"summary.json não encontrado em {run}")
        with summary_path.open("r", encoding="utf-8") as f:
            summaries.append((run, json.load(f)))

    base_name, base_summary = summaries[0]
    metrics = [
        "return_net",
        "max_drawdown",
        "profit_factor",
        "sharpe",
        "win_rate",
        "expectancy",
        "turnover",
        "avg_hours_in_pos",
        "total_trades",
    ]

    table = Table(title=f"Compare Runs (base: {Path(base_name).name})")
    table.add_column("run")
    for metric in metrics:
        table.add_column(metric)
        table.add_column(f"Δ {metric}")

    compare_payload: dict[str, Any] = {"base_run": base_name, "results": []}

    for run_name, summary in summaries[1:]:
        row = [Path(run_name).name]
        deltas: dict[str, Any] = {}
        for metric in metrics:
            cur_val = float(summary.get(metric, 0.0))
            base_val = float(base_summary.get(metric, 0.0))
            delta = cur_val - base_val
            deltas[metric] = {"value": cur_val, "delta": delta}
            row.append(f"{cur_val:.6f}")
            row.append(f"{delta:+.6f}")
        table.add_row(*row)
        compare_payload["results"].append({"run": run_name, "metrics": deltas})

    Console().print(table)
    if save_path:
        write_json(Path(save_path), compare_payload)
        print(f"[green]Comparação salva em:[/green] {save_path}")


@app.command("grid")
def grid(
    data_path: str = typer.Option(..., "--data-path"),
    config: str = typer.Option("config/settings.yaml", "--config"),
    outdir: str = typer.Option("runs", "--outdir"),
    param: list[str] = typer.Option([], "--param"),
    tag: str | None = typer.Option(None, "--tag"),
    seed: int | None = typer.Option(None, "--seed"),
):
    cfg = load_settings(config)
    base_dict = cfg.model_dump(mode="json")

    parsed_params = [_parse_param_option(p) for p in param]
    keys = [k for k, _ in parsed_params]
    value_lists = [values for _, values in parsed_params]

    combinations = list(itertools.product(*value_lists)) if value_lists else [()]
    print(f"[cyan]Grid:[/cyan] {len(combinations)} combinação(ões)")

    manager = RunManager(outdir)

    for combo in combinations:
        cfg_dict = json.loads(json.dumps(base_dict))
        name_tokens = []
        for key, value in zip(keys, combo):
            _set_nested_value(cfg_dict, key, value)
            name_tokens.append(f"{key.split('.')[-1]}-{value}")

        run_cfg = Settings.model_validate(cfg_dict)
        params_hash = RunManager.stable_hash_from_settings(run_cfg)
        combo_name = "_".join(name_tokens) if name_tokens else "base"
        combo_name = combo_name.replace("/", "-").replace(" ", "")
        auto_name = RunManager.auto_run_name(run_cfg, params_hash)
        run_name = f"{auto_name}_{combo_name}"
        ctx = manager.build_context(run_cfg, run_name=run_name)

        manager.init_run(ctx, settings=run_cfg, data_path=data_path, config_path=config, seed=seed, tag=tag)
        trades, equity, summary, regime_stats, direction_stats = _execute_backtest(data_path, run_cfg)
        manager.persist_outputs(
            ctx,
            summary=summary,
            trades=trades,
            equity=equity,
            regime_stats=regime_stats,
            direction_stats=direction_stats,
        )
        print(f"[green]Grid run concluída:[/green] {ctx.run_dir}")


@app.command("paper")
def paper(loop: bool = False, sleep: int = 60, data_path: str = typer.Option(""), config: str = "config/settings.yaml"):
    cfg = load_settings(config)

    def run_once() -> None:
        if data_path:
            df = load_parquet(data_path)
        else:
            df = BinanceDataClient().fetch_ohlcv(cfg.symbol, cfg.interval, cfg.start_date, cfg.end_date)
        df = build_features(df)
        df["regime"] = RegimeDetector(cfg.regime).apply(df)
        trades, equity = BacktestEngine(cfg).run(df)
        last_equity = float(equity["equity"].iloc[-1]) if not equity.empty else cfg.risk.account_equity_usdt
        logger.info("paper_cycle", extra={"extra": {"trades": len(trades), "equity": last_equity}})

    if not loop:
        run_once()
        return
    while True:
        run_once()
        time.sleep(sleep)


@app.command("live")
def live(dry_run: bool = typer.Option(True), config: str = "config/settings.yaml"):
    cfg = load_settings(config)
    broker = BrokerBinance(cfg.execution)
    perms = broker.validate_permissions()
    logger.info("api_permissions", extra={"extra": {"permissions": perms}})

    if dry_run:
        logger.info("live_dry_run", extra={"extra": {"action": "would_fetch_state_and_place_orders"}})
        return

    state = broker.get_account_state()
    logger.info("account_state", extra={"extra": {"state": state}})


if __name__ == "__main__":
    app()
