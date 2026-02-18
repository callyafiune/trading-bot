from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RegimeSettings(BaseModel):
    vol_window_short: int = 24
    vol_window_long: int = 168
    chaos_vol_percentile: float = 0.9
    adx_period: int = 14
    adx_trend_threshold: float = 28
    adx_enter_threshold: float = 28
    adx_exit_threshold: float = 24
    macro_slope_min: float = 0.0
    macro_confirm_bars: int = 6
    micro_confirm_bars: int = 3
    chaos_vol_threshold: float = 0.03
    bb_period: int = 20
    bb_width_range_threshold: float = 0.06
    chaos_atr_pct_threshold: float = 0.03


class StrategyRouterSettings(BaseModel):
    class RegimeOverride(BaseModel):
        breakout_N: int

    class RouterOverrides(BaseModel):
        bull_trend: "StrategyRouterSettings.RegimeOverride" = Field(
            default_factory=lambda: StrategyRouterSettings.RegimeOverride(breakout_N=120)
        )
        bear_trend: "StrategyRouterSettings.RegimeOverride" = Field(
            default_factory=lambda: StrategyRouterSettings.RegimeOverride(breakout_N=72)
        )

    enable_trend_up_long: bool = True
    enable_trend_down_short: bool = True
    enable_range: bool = False
    enable_chaos: bool = False
    short_use_ma200_filter: bool = False
    bull_slope_min: float = 0.0
    cooldown_bars_after_exit: int = 6
    overrides: RouterOverrides = Field(default_factory=RouterOverrides)


class RouterSettings(BaseModel):
    enabled: bool = False
    chaos_trade: bool = False
    range_risk_multiplier: float = 0.5
    trend_mode: Literal["breakout", "baseline", "ema", "ema_macd", "ml_gate"] = "breakout"
    range_mode: Literal["breakout", "baseline", "ema", "ema_macd", "ml_gate"] = "ema_macd"


class StrategyBreakoutSettings(BaseModel):
    class RetestSettings(BaseModel):
        enabled: bool = False
        window_bars: int = 6
        tolerance_atr: float = 0.25
        confirmation: Literal["close_back", "wick_reject"] = "close_back"

    mode: Literal["breakout", "baseline", "ema", "ema_macd", "ml_gate"] = "breakout"
    breakout_lookback_N: int = 72
    ema_fast_period: int = 12
    ema_slow_period: int = 26
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    atr_period: int = 14
    atr_k: float = 2.5
    use_ma200_filter: bool = True
    ma200_period: int = 200
    min_rel_volume: float = 1.0
    use_rel_volume_filter: bool = True
    time_stop_hours: int = 48
    use_walk_forward: bool = True
    wf_train_bars: int = 24 * 60
    wf_val_bars: int = 24 * 30
    wf_test_bars: int = 24 * 30
    ml_prob_threshold: float = 0.55
    ml_feature_selector: Literal["lightgbm", "xgboost"] = "lightgbm"
    ml_feature_top_k: int = 8
    trade_direction: Literal["both", "long", "short"] = "both"
    retest: RetestSettings = Field(default_factory=RetestSettings)


class RiskSettings(BaseModel):
    account_equity_usdt: float = 10000
    risk_per_trade: float = 0.005
    max_leverage: float = 2.0
    max_open_positions: int = 1
    daily_loss_limit_pct: float = 0.02
    weekly_loss_limit_pct: float = 0.05
    liquidation_buffer_pct: float = 0.35


class FrictionsSettings(BaseModel):
    fee_rate_per_side: float = 0.001
    slippage_rate_per_side: float = 0.0003
    borrow_interest_rate_per_hour: float = 0.00002


class ExecutionSettings(BaseModel):
    class DetailTimeframeSettings(BaseModel):
        enabled: bool = False
        timeframe: str = "15m"
        policy: Literal["conservative", "optimistic"] = "conservative"

    margin_mode: Literal["isolated", "cross"] = "isolated"
    order_type: Literal["MARKET"] = "MARKET"
    use_testnet: bool = True
    api_key_env: str = "BINANCE_API_KEY"
    api_secret_env: str = "BINANCE_API_SECRET"
    dry_run: bool = True
    detail_timeframe: DetailTimeframeSettings = Field(default_factory=DetailTimeframeSettings)


class FeatureSettings(BaseModel):
    ema_fast: int = 50
    ema_slow: int = 200
    slope_window: int = 10
    vol_window: int = 48
    annualize_vol: bool = False


class FundingFilterSettings(BaseModel):
    enabled: bool = False
    z_window: int = 168
    z_threshold: float = 1.0
    block_short_if_z_lt: float = -1.0
    block_long_if_z_gt: float = 1.0


class FngFilterSettings(BaseModel):
    enabled: bool = False
    path: str = ""
    block_long_if_gte: int = 80
    block_short_if_lte: int = 20


class MultiTimeframeSettings(BaseModel):
    enabled: bool = False
    timeframe: str = "4h"
    ma_period: int = 200
    require_trend_alignment: bool = True


class TimeExitSettings(BaseModel):
    enabled: bool = False
    max_holding_hours: int = 24
    soft_exit_hours: int = 12
    min_r_multiple_after_soft: float = 1.0


class AdaptiveTrailingSettings(BaseModel):
    enabled: bool = False
    activate_after_R: float = 1.5
    trailing_atr_multiplier: float = 1.5


class Settings(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    start_date: str
    end_date: str
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    regime: RegimeSettings = Field(default_factory=RegimeSettings)
    funding_filter: FundingFilterSettings = Field(default_factory=FundingFilterSettings)
    fng_filter: FngFilterSettings = Field(default_factory=FngFilterSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    multi_timeframe: MultiTimeframeSettings = Field(default_factory=MultiTimeframeSettings)
    time_exit: TimeExitSettings = Field(default_factory=TimeExitSettings)
    adaptive_trailing: AdaptiveTrailingSettings = Field(default_factory=AdaptiveTrailingSettings)
    strategy_breakout: StrategyBreakoutSettings = Field(default_factory=StrategyBreakoutSettings)
    strategy_router: StrategyRouterSettings = Field(default_factory=StrategyRouterSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    frictions: FrictionsSettings = Field(default_factory=FrictionsSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Settings.model_validate(data)
