import pandas as pd

from bot.runs.run_manager import MIN_TRADE_COLUMNS, RunManager
from bot.utils.config import load_settings


def test_config_hash_stable():
    cfg = load_settings()
    h1 = RunManager.stable_hash_from_settings(cfg)
    h2 = RunManager.stable_hash_from_settings(cfg)
    assert h1 == h2


def test_run_directory_and_main_artifacts_created(tmp_path):
    cfg = load_settings()
    manager = RunManager(tmp_path)
    ctx = manager.build_context(cfg, run_name="test_run")
    manager.init_run(ctx, settings=cfg, data_path="data/sample.parquet", config_path="config/settings.yaml", seed=42, tag="unit")

    trades = pd.DataFrame(
        [
            {
                "trade_id": 1,
                "direction": "LONG",
                "entry_time": "2024-01-01T00:00:00Z",
                "entry_price": 100.0,
                "exit_time": "2024-01-01T01:00:00Z",
                "exit_price": 101.0,
                "qty": 1.0,
                "notional": 100.0,
                "stop_init": 95.0,
                "stop_final": 96.0,
                "reason_exit": "TIME",
                "pnl_gross": 1.0,
                "pnl_net": 0.8,
                "fees": 0.1,
                "slippage": 0.05,
                "interest": 0.05,
                "regime_at_entry": "TREND",
                "hold_hours": 1,
            }
        ]
    )
    equity = pd.DataFrame(
        [
            {"timestamp": "2024-01-01T00:00:00Z", "equity": 10000.0, "position": "FLAT", "price": 100.0, "drawdown": 0.0},
            {"timestamp": "2024-01-01T01:00:00Z", "equity": 10000.8, "position": "FLAT", "price": 101.0, "drawdown": 0.0},
        ]
    )
    summary = {
        "return_net": 0.001,
        "max_drawdown": 0.0,
        "profit_factor": 1.5,
        "sharpe": 0.2,
        "win_rate": 1.0,
        "expectancy": 0.8,
        "turnover": 0.01,
        "avg_hours_in_pos": 1.0,
        "total_trades": 1,
        "start_ts": "2024-01-01T00:00:00Z",
        "end_ts": "2024-01-01T01:00:00Z",
        "fees_total": 0.1,
        "slippage_total": 0.05,
        "interest_total": 0.05,
        "long_trades": 1,
        "short_trades": 0,
        "pnl_long": 0.8,
        "pnl_short": 0.0,
        "counts": {
            "signals_total": 1,
            "entries_executed": 1,
            "blocked_regime": 0,
            "blocked_risk": 0,
            "blocked_killswitch": 0,
            "killswitch_events": 0,
        },
    }

    manager.persist_outputs(
        ctx,
        summary=summary,
        trades=trades,
        equity=equity,
        regime_stats={"TREND": {"candles_count": 1, "pct_time": 1.0, "trades_count": 1, "pnl_net_total": 0.8, "avg_pnl": 0.8, "win_rate": 1.0}},
        direction_stats={"LONG": {"trades_count": 1, "pnl_net_total": 0.8, "avg_pnl": 0.8, "win_rate": 1.0, "avg_hold_hours": 1.0}},
    )

    required = [
        "config_used.yaml",
        "summary.json",
        "trades.csv",
        "equity.csv",
        "regime_stats.json",
        "direction_stats.json",
        "params_hash.txt",
        "run_meta.json",
    ]
    for file_name in required:
        assert (ctx.run_dir / file_name).exists()


def test_trades_csv_has_minimum_columns(tmp_path):
    cfg = load_settings()
    manager = RunManager(tmp_path)
    ctx = manager.build_context(cfg, run_name="columns_run")
    manager.init_run(ctx, settings=cfg, data_path="data/sample.parquet", config_path="config/settings.yaml")

    trades = pd.DataFrame([{"trade_id": 1, "direction": "LONG", "entry_price": 100.0, "exit_price": 101.0, "pnl_net": 1.0}])
    manager.persist_outputs(
        ctx,
        summary={
            "return_net": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "turnover": 0.0,
            "avg_hours_in_pos": 0.0,
            "total_trades": 0,
            "start_ts": None,
            "end_ts": None,
            "fees_total": 0.0,
            "slippage_total": 0.0,
            "interest_total": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "pnl_long": 0.0,
            "pnl_short": 0.0,
            "counts": {
                "signals_total": 0,
                "entries_executed": 0,
                "blocked_regime": 0,
                "blocked_risk": 0,
                "blocked_killswitch": 0,
                "killswitch_events": 0,
            },
        },
        trades=trades,
        equity=pd.DataFrame([{"timestamp": "2024-01-01T00:00:00Z", "equity": 10000, "position": "FLAT", "price": 100.0, "drawdown": 0.0}]),
        regime_stats={},
        direction_stats={},
    )

    df_saved = pd.read_csv(ctx.run_dir / "trades.csv")
    assert list(df_saved.columns) == MIN_TRADE_COLUMNS
