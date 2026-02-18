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
from bot.market_data.loader import (
    derive_detail_data_path,
    load_parquet,
    merge_fng_with_ohlcv,
    merge_ohlcv_with_funding,
    process_funding_to_1h,
    resample_ohlcv,
    save_parquet,
)
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
    summary["trades_by_regime"] = trades.groupby("regime_at_entry").size().astype(int).to_dict() if not trades.empty else {}
    summary["pnl_by_regime"] = trades.groupby("regime_at_entry")["pnl_net"].sum().astype(float).to_dict() if not trades.empty else {}
    summary["trades_by_regime_final"] = summary["trades_by_regime"]
    summary["pnl_by_regime_final"] = summary["pnl_by_regime"]
    for regime_name in ["BULL_RANGE", "BEAR_RANGE"]:
        summary["trades_by_regime_final"].setdefault(regime_name, 0)
        summary["pnl_by_regime_final"].setdefault(regime_name, 0.0)
    summary["blocked_by_regime_reason"] = diagnostics.get("blocked_by_regime_reason", {})
    summary["regime_switch_count_macro"] = int(diagnostics.get("regime_switch_count_macro", 0))
    summary["regime_switch_count_micro"] = int(diagnostics.get("regime_switch_count_micro", 0))
    summary["regime_switch_count_total"] = int(diagnostics.get("regime_switch_count_total", 0))
    summary["blocked_funding"] = int(diagnostics.get("blocked_funding", 0))
    summary["blocked_macro"] = int(diagnostics.get("blocked_macro", 0))
    summary["blocked_micro"] = int(diagnostics.get("blocked_micro", 0))
    summary["blocked_chaos"] = int(diagnostics.get("blocked_chaos", 0))
    summary["blocked_range_flat"] = int(diagnostics.get("blocked_by_regime_reason", {}).get("blocked_range_flat", 0))
    summary["blocked_cooldown"] = int(diagnostics.get("blocked_cooldown", 0))
    summary["mtf_enabled"] = bool(diagnostics.get("mtf_enabled", False))
    summary["time_exit_triggered"] = int(diagnostics.get("time_exit_triggered", 0))
    summary["adaptive_trailing_triggered"] = int(diagnostics.get("adaptive_trailing_triggered", 0))
    summary["adaptive_trailing_stop_hits"] = int(diagnostics.get("adaptive_trailing_stop_hits", 0))
    summary["detail_timeframe_enabled"] = bool(diagnostics.get("detail_timeframe_enabled", False))
    summary["detail_timeframe"] = diagnostics.get("detail_timeframe", None)
    summary["detail_policy"] = diagnostics.get("detail_policy", None)
    summary["biggest_winner_pct"] = float((trades["pnl_net"] / trades["notional"]).max()) if not trades.empty else 0.0
    summary["worst_loser_pct"] = float((trades["pnl_net"] / trades["notional"]).min()) if not trades.empty else 0.0
    summary["max_trade_R"] = float(trades["R_multiple_exit"].max()) if "R_multiple_exit" in trades.columns and not trades.empty else 0.0
    summary["entry_direct_count"] = int((trades.get("entry_type", pd.Series(dtype=str)) == "direct").sum()) if not trades.empty else 0
    summary["entry_retest_count"] = int((trades.get("entry_type", pd.Series(dtype=str)) == "retest").sum()) if not trades.empty else 0
    if not trades.empty:
        dir_stats = {}
        for direction in ["LONG", "SHORT"]:
            td = trades.loc[trades["direction"] == direction]
            gp = td.loc[td["pnl_net"] > 0, "pnl_net"].sum()
            gl = td.loc[td["pnl_net"] < 0, "pnl_net"].sum()
            dir_stats[direction] = {
                "trades": int(len(td)),
                "return_net": float(td["pnl_net"].sum() / max(equity["equity"].iloc[0], 1e-9)) if not equity.empty else 0.0,
                "profit_factor": float(gp / abs(gl)) if gl < 0 else 0.0,
            }
        summary["direction_buckets"] = dir_stats

        regime_stats = {}
        for regime, rg in trades.groupby("regime_at_entry"):
            gp = rg.loc[rg["pnl_net"] > 0, "pnl_net"].sum()
            gl = rg.loc[rg["pnl_net"] < 0, "pnl_net"].sum()
            regime_stats[str(regime)] = {
                "trades": int(len(rg)),
                "return_net": float(rg["pnl_net"].sum() / max(equity["equity"].iloc[0], 1e-9)) if not equity.empty else 0.0,
                "profit_factor": float(gp / abs(gl)) if gl < 0 else 0.0,
            }
        summary["regime_buckets"] = regime_stats
    else:
        summary["direction_buckets"] = {}
        summary["regime_buckets"] = {}
    summary["counts"] = {
        "signals_total": int(diagnostics.get("signals_total", 0)),
        "entries_executed": int(diagnostics.get("entries_executed", 0)),
        "blocked_regime": int(diagnostics.get("signals_blocked_regime", 0)),
        "blocked_risk": int(diagnostics.get("signals_blocked_risk", 0)),
        "blocked_killswitch": int(diagnostics.get("signals_blocked_killswitch", 0)),
        "blocked_mode": int(diagnostics.get("signals_blocked_mode", 0)),
        "blocked_ma200": int(diagnostics.get("signals_blocked_ma200", 0)),
        "killswitch_events": int(diagnostics.get("killswitch_events", 0)),
        "blocked_funding": int(diagnostics.get("blocked_funding", 0)),
        "blocked_macro": int(diagnostics.get("blocked_macro", 0)),
        "blocked_micro": int(diagnostics.get("blocked_micro", 0)),
        "blocked_chaos": int(diagnostics.get("blocked_chaos", 0)),
        "blocked_range_flat": int(diagnostics.get("blocked_by_regime_reason", {}).get("blocked_range_flat", 0)),
        "blocked_cooldown": int(diagnostics.get("blocked_cooldown", 0)),
        "blocked_mtf": int(diagnostics.get("blocked_mtf", 0)),
        "blocked_funding_count": int(diagnostics.get("blocked_funding", 0)),
    }
    return summary


def _attach_mtf_features(df_1h: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    if not cfg.multi_timeframe.enabled:
        return df_1h
    if cfg.multi_timeframe.timeframe.lower() != "4h":
        raise typer.BadParameter("Apenas timeframe 4h é suportado em multi_timeframe.timeframe no momento.")

    df_4h = resample_ohlcv(df_1h, "4h")
    ema_period = int(cfg.multi_timeframe.ma_period)
    df_4h["ema_200_4h"] = df_4h["close"].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    df_4h["ema_slope_4h"] = df_4h["ema_200_4h"].diff()

    mtf = df_4h[["open_time", "close", "ema_200_4h", "ema_slope_4h"]].rename(columns={"close": "close_4h"})
    mtf = mtf.set_index("open_time").shift(1)
    aligned = mtf.reindex(df_1h.set_index("open_time").index, method="ffill")
    aligned = aligned.reset_index()
    return df_1h.merge(aligned, on="open_time", how="left")


def _build_regime_stats(df: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    total_candles = len(df)
    regimes = sorted(df["regime"].dropna().astype(str).unique().tolist()) if "regime" in df.columns else []
    result: dict[str, Any] = {}

    for regime in regimes:
        candles_count = int((df["regime"] == regime).sum())
        regime_trades = trades.loc[trades["regime_at_entry"] == regime] if not trades.empty else pd.DataFrame()
        trades_count = int(len(regime_trades))
        pnl_total = float(regime_trades["pnl_net"].sum()) if trades_count else 0.0
        result[regime] = {
            "candles_count": candles_count,
            "pct_time": float(candles_count / total_candles) if total_candles else 0.0,
            "trades_count": trades_count,
            "pnl_net_total": pnl_total,
        }

    result["switches"] = {
        "macro": int(df.get("regime_switch_macro", pd.Series(dtype=bool)).sum()),
        "micro": int(df.get("regime_switch_micro", pd.Series(dtype=bool)).sum()),
        "total": int(df.get("regime_switch_total", pd.Series(dtype=bool)).sum()),
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
    funding_path: str | None = None,
    detail_data_path: str | None = None,
    fng_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    df = load_parquet(data_path)
    if cfg.funding_filter.enabled and funding_path is None:
        data_name = Path(data_path).name
        if data_name.startswith("BTCUSDT_1h_"):
            candidate = Path(data_path).with_name(f"funding_BTCUSDT_1h_{data_name.split('BTCUSDT_1h_', 1)[1]}")
            if candidate.exists():
                funding_path = str(candidate)
        if funding_path is None:
            print("[red]funding_filter.enabled=true requer --funding-path (ou arquivo derivado existente).[/red]")
            raise typer.Exit(code=2)

    if funding_path:
        funding_df = load_parquet(funding_path)
        df = merge_ohlcv_with_funding(df, funding_df)

    if cfg.risk_fng.enabled:
        resolved_fng = fng_path or cfg.risk_fng.path
        if resolved_fng:
            fng_df = load_parquet(resolved_fng)
            df = merge_fng_with_ohlcv(df, fng_df)

    if df.empty:
        print("[red]Dataset vazio. Verifique o arquivo de entrada.[/red]")
        raise typer.Exit(code=1)

    print(
        "[cyan]Dataset:[/cyan]"
        f" candles={len(df)}"
        f" | inicio={df['open_time'].min()}"
        f" | fim={df['open_time'].max()}"
    )

    df = _attach_mtf_features(df, cfg)
    df = build_features(df, cfg.features)
    df["regime"] = RegimeDetector(cfg.regime).apply(df)
    trend_up = int(df["regime"].astype(str).str.startswith("BULL").sum())
    trend_down = int(df["regime"].astype(str).str.startswith("BEAR").sum())
    trend_pct = ((trend_up + trend_down) / len(df)) * 100 if len(df) else 0.0
    print(f"[cyan]Regime:[/cyan] BULL={trend_up} | BEAR={trend_down} ({trend_pct:.2f}% macro trend)")

    detail_df: pd.DataFrame | None = None
    if cfg.execution.detail_timeframe.enabled:
        resolved_detail_path = detail_data_path
        if not resolved_detail_path:
            candidate = derive_detail_data_path(data_path, cfg.execution.detail_timeframe.timeframe)
            if candidate and candidate.exists():
                resolved_detail_path = str(candidate)
        if resolved_detail_path:
            detail_df = load_parquet(resolved_detail_path)
            print(
                f"[cyan]Detail TF:[/cyan] enabled={cfg.execution.detail_timeframe.enabled}"
                f" | timeframe={cfg.execution.detail_timeframe.timeframe}"
                f" | policy={cfg.execution.detail_timeframe.policy}"
                f" | candles={len(detail_df)}"
            )
        else:
            print("[yellow]Detail timeframe habilitado, mas dataset não encontrado. Rodando em HTF-only.[/yellow]")

    engine = BacktestEngine(cfg)
    trades, equity = engine.run(df, detail_df=detail_df)
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

    engine.last_run_diagnostics["regime_switch_count_macro"] = int(df.get("regime_switch_macro", pd.Series(dtype=bool)).sum())
    engine.last_run_diagnostics["regime_switch_count_micro"] = int(df.get("regime_switch_micro", pd.Series(dtype=bool)).sum())
    engine.last_run_diagnostics["regime_switch_count_total"] = int(df.get("regime_switch_total", pd.Series(dtype=bool)).sum())
    engine.last_run_diagnostics["mtf_enabled"] = bool(cfg.multi_timeframe.enabled)
    engine.last_run_diagnostics["detail_timeframe"] = cfg.execution.detail_timeframe.timeframe if cfg.execution.detail_timeframe.enabled else None
    engine.last_run_diagnostics["detail_policy"] = cfg.execution.detail_timeframe.policy if cfg.execution.detail_timeframe.enabled else None

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


@app.command("fetch-funding")
def fetch_funding(
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    symbol: str = typer.Option("BTCUSDT", "--symbol"),
):
    client = BinanceDataClient()
    funding_df = client.fetch_funding_rate(symbol=symbol, start_date=start, end_date=end)

    raw_path = Path(f"data/raw/funding/{symbol}_funding_{start}_{end}.parquet")
    proc_path = Path(f"data/processed/funding_{symbol}_1h_{start}_{end}.parquet")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    proc_path.parent.mkdir(parents=True, exist_ok=True)

    funding_df.to_parquet(raw_path, index=False)
    funding_1h = process_funding_to_1h(funding_df, start, end)
    funding_1h.to_parquet(proc_path, index=False)
    print(f"Funding salvo em {raw_path} e {proc_path}")


@app.command("fetch-fng")
def fetch_fng(
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
):
    client = BinanceDataClient()
    fng_df = client.fetch_fear_greed(start, end)
    raw_path = Path(f"data/raw/fng/fng_1d_{start}_{end}.parquet")
    proc_path = Path(f"data/processed/fng_1d_{start}_{end}.parquet")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    fng_df.to_parquet(raw_path, index=False)
    fng_df.to_parquet(proc_path, index=False)
    print(f"Fear & Greed salvo em {raw_path} e {proc_path}")


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
    funding_path: str | None = typer.Option(None, "--funding-path"),
    detail_data_path: str | None = typer.Option(None, "--detail-data-path"),
    detail_timeframe: str | None = typer.Option(None, "--detail-timeframe"),
    no_detail_timeframe: bool = typer.Option(False, "--no-detail-timeframe"),
    retest: bool | None = typer.Option(None, "--require-retest/--no-require-retest"),
    retest_window: int | None = typer.Option(None, "--retest-window"),
    retest_tolerance_atr: float | None = typer.Option(None, "--retest-tolerance-atr"),
    use_ma200_filter: bool | None = typer.Option(None, "--use-ma200-filter/--no-use-ma200-filter"),
    use_mtf: bool | None = typer.Option(None, "--enable-mtf/--disable-mtf"),
    fng_path: str | None = typer.Option(None, "--fng-path"),
    ml_threshold: float | None = typer.Option(None, "--ml-threshold"),
    long_only: bool = typer.Option(False, "--long-only"),
    short_only: bool = typer.Option(False, "--short-only"),
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
        cfg.regime.adx_enter_threshold = adx_threshold
        cfg.regime.adx_exit_threshold = max(10.0, adx_threshold - 4.0)
        cfg.regime.adx_trend_threshold = adx_threshold
    if use_ma200_filter is not None:
        cfg.strategy_breakout.use_ma200_filter = use_ma200_filter
    if use_mtf is not None:
        cfg.multi_timeframe.enabled = use_mtf
    if detail_timeframe is not None:
        cfg.execution.detail_timeframe.enabled = True
        cfg.execution.detail_timeframe.timeframe = detail_timeframe
    if no_detail_timeframe:
        cfg.execution.detail_timeframe.enabled = False
    if retest is not None:
        cfg.strategy_breakout.retest.enabled = retest
    if retest_window is not None:
        cfg.strategy_breakout.retest.window_bars = retest_window
    if retest_tolerance_atr is not None:
        cfg.strategy_breakout.retest.tolerance_atr = retest_tolerance_atr
    if ml_threshold is not None:
        cfg.strategy_breakout.ml_prob_threshold = ml_threshold
    if long_only and short_only:
        raise typer.BadParameter("Use apenas uma entre --long-only e --short-only")
    if long_only:
        cfg.strategy_breakout.trade_direction = "long"
    if short_only:
        cfg.strategy_breakout.trade_direction = "short"
    manager = RunManager(outdir)
    ctx = manager.build_context(cfg, run_name=run_name)

    if dry_run:
        print(f"[yellow]DRY RUN[/yellow] run_dir={ctx.run_dir} params_hash={ctx.params_hash}")
        return

    manager.init_run(ctx, settings=cfg, data_path=data_path, config_path=config, seed=seed, tag=tag)
    trades, equity, summary, regime_stats, direction_stats = _execute_backtest(
        data_path,
        cfg,
        funding_path=funding_path,
        detail_data_path=detail_data_path,
        fng_path=fng_path,
    )
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
    runs: list[str] = typer.Option(..., "--runs"),
    extra_runs: list[str] = typer.Argument([]),
    save_path: str | None = typer.Option(None, "--save-path"),
):
    all_runs = [*runs, *(extra_runs or [])]
    if len(all_runs) < 2:
        raise typer.BadParameter("Informe ao menos duas runs em --runs")

    summaries: list[tuple[str, dict[str, Any]]] = []
    for run in all_runs:
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
        "biggest_winner_pct",
        "worst_loser_pct",
        "max_trade_R",
        "time_exit_triggered",
        "adaptive_trailing_triggered",
        "blocked_funding",
    ]

    table = Table(title=f"Compare Runs (base: {Path(base_name).name})")
    table.add_column("run")
    for metric in metrics:
        table.add_column(metric)
        table.add_column(f"delta_{metric}")

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
        df = build_features(df, cfg.features)
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
