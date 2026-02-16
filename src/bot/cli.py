from __future__ import annotations

import time
from pathlib import Path

import typer
from rich import print

from bot.backtest.engine import BacktestEngine
from bot.backtest.metrics import compute_metrics
from bot.backtest.reporting import print_summary, save_trades_csv
from bot.execution.broker_binance import BrokerBinance
from bot.features.builder import build_features
from bot.market_data.binance_client import BinanceDataClient
from bot.market_data.loader import load_parquet, save_parquet
from bot.regime.detector import RegimeDetector
from bot.utils.config import load_settings
from bot.utils.logging import setup_logger

app = typer.Typer()
logger = setup_logger()


@app.command("fetch-data")
def fetch_data(start: str = typer.Option(None), end: str = typer.Option(None), config_path: str = "config/settings.yaml"):
    cfg = load_settings(config_path)
    start_date = start or cfg.start_date
    end_date = end or cfg.end_date
    df = BinanceDataClient().fetch_ohlcv(cfg.symbol, cfg.interval, start_date, end_date)
    raw_path = Path(f"data/raw/{cfg.symbol}_{cfg.interval}_{start_date}_{end_date}.parquet")
    proc_path = Path(f"data/processed/{cfg.symbol}_{cfg.interval}_{start_date}_{end_date}.parquet")
    save_parquet(df, raw_path, proc_path, cfg.interval)
    print(f"Dados salvos em {raw_path} e {proc_path}")


@app.command("backtest")
def backtest(data_path: str = typer.Option(...), config_path: str = "config/settings.yaml"):
    cfg = load_settings(config_path)
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

    trades, equity = BacktestEngine(cfg).run(df)
    print(f"[cyan]Backtest:[/cyan] total_trades={len(trades)}")

    save_trades_csv(trades)
    metrics = compute_metrics(trades, equity)
    print_summary(metrics)


@app.command("paper")
def paper(loop: bool = False, sleep: int = 60, data_path: str = typer.Option(""), config_path: str = "config/settings.yaml"):
    cfg = load_settings(config_path)

    def run_once() -> None:
        if data_path:
            df = load_parquet(data_path)
        else:
            df = BinanceDataClient().fetch_ohlcv(cfg.symbol, cfg.interval, cfg.start_date, cfg.end_date)
        df = build_features(df)
        df["regime"] = RegimeDetector(cfg.regime).apply(df)
        trades, equity = BacktestEngine(cfg).run(df)
        logger.info("paper_cycle", extra={"extra": {"trades": len(trades), "equity": float(equity.iloc[-1]) if len(equity) else cfg.risk.account_equity_usdt}})

    if not loop:
        run_once()
        return
    while True:
        run_once()
        time.sleep(sleep)


@app.command("live")
def live(dry_run: bool = typer.Option(True), config_path: str = "config/settings.yaml"):
    cfg = load_settings(config_path)
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
