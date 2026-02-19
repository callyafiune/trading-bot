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
    blocked_funding_count: int = 0
    blocked_funding_long: int = 0
    blocked_funding_short: int = 0
    blocked_fng_long: int = 0
    blocked_fng_short: int = 0
    blocked_structure_total: int = 0
    blocked_structure_long: int = 0
    blocked_structure_short: int = 0


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
    blocked_funding_long: int = 0
    blocked_funding_short: int = 0
    blocked_funding_total: int = 0
    blocked_fng_long: int = 0
    blocked_fng_short: int = 0
    blocked_fng_total: int = 0
    blocked_structure_total: int = 0
    blocked_structure_long: int = 0
    blocked_structure_short: int = 0
    msb_bull_count: int = 0
    msb_bear_count: int = 0
    trades_taken_after_msb: int = 0
    trades_taken_in_bull_structure: int = 0
    trades_taken_in_bear_structure: int = 0
    trades_taken_in_neutral: int = 0
    blocked_macro: int = 0
    blocked_micro: int = 0
    blocked_chaos: int = 0
    blocked_range_flat: int = 0
    blocked_cooldown: int = 0
    mtf_enabled: bool = False
    detail_timeframe_enabled: bool = False
    detail_timeframe: str | None = None
    detail_policy: str | None = None
    time_exit_triggered: int = 0
    adaptive_trailing_triggered: int = 0
    adaptive_trailing_stop_hits: int = 0
    biggest_winner_pct: float = 0.0
    worst_loser_pct: float = 0.0
    max_trade_R: float = 0.0
    entry_direct_count: int = 0
    entry_retest_count: int = 0
    direction_buckets: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    regime_buckets: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    features_present: list[str] = Field(default_factory=list)
    time_in_regime: dict[str, int] = Field(default_factory=dict)
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
