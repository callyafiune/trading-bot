from __future__ import annotations

from pydantic import BaseModel, Field


class SummaryCounts(BaseModel):
    signals_total: int = 0
    entries_executed: int = 0
    blocked_regime: int = 0
    blocked_risk: int = 0
    blocked_killswitch: int = 0
    blocked_mode: int = 0
    blocked_ma200: int = 0
    killswitch_events: int = 0
    blocked_funding: int = 0
    blocked_macro: int = 0
    blocked_micro: int = 0
    blocked_chaos: int = 0
    blocked_range_flat: int = 0
    blocked_cooldown: int = 0
    blocked_mtf: int = 0


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
    trades_by_regime: dict[str, int] = Field(default_factory=dict)
    pnl_by_regime: dict[str, float] = Field(default_factory=dict)
    trades_by_regime_final: dict[str, int] = Field(default_factory=dict)
    pnl_by_regime_final: dict[str, float] = Field(default_factory=dict)
    blocked_by_regime_reason: dict[str, int] = Field(default_factory=dict)
    regime_switch_count_macro: int = 0
    regime_switch_count_micro: int = 0
    regime_switch_count_total: int = 0
    blocked_funding: int = 0
    blocked_macro: int = 0
    blocked_micro: int = 0
    blocked_chaos: int = 0
    blocked_range_flat: int = 0
    blocked_cooldown: int = 0
    mtf_enabled: bool = False
    time_exit_triggered: int = 0
    adaptive_trailing_triggered: int = 0
    adaptive_trailing_stop_hits: int = 0
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
