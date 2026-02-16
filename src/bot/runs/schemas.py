from __future__ import annotations

from pydantic import BaseModel, Field


class SummaryCounts(BaseModel):
    signals_total: int = 0
    entries_executed: int = 0
    blocked_regime: int = 0
    blocked_risk: int = 0
    blocked_killswitch: int = 0
    blocked_mode: int = 0
    killswitch_events: int = 0


class BacktestSummary(BaseModel):
    return_net: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    expectancy: float = 0.0
    turnover: float = 0.0
    avg_hours_in_pos: float = 0.0
    total_trades: int = 0
    start_ts: str | None = None
    end_ts: str | None = None
    fees_total: float = 0.0
    slippage_total: float = 0.0
    interest_total: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    pnl_long: float = 0.0
    pnl_short: float = 0.0
    counts: SummaryCounts = Field(default_factory=SummaryCounts)


class RunMeta(BaseModel):
    run_id: str
    run_name: str
    created_at: str
    outdir: str
    data_path: str
    config_path: str
    git_commit: str | None = None
    python_version: str
    seed: int | None = None
    tag: str | None = None
    params_hash: str
