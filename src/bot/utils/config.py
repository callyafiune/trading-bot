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
    bb_period: int = 20
    bb_width_range_threshold: float = 0.06


class StrategyBreakoutSettings(BaseModel):
    mode: Literal["breakout", "ema", "ema_macd", "ml_gate"] = "breakout"
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
    margin_mode: Literal["isolated", "cross"] = "isolated"
    order_type: Literal["MARKET"] = "MARKET"
    use_testnet: bool = True
    api_key_env: str = "BINANCE_API_KEY"
    api_secret_env: str = "BINANCE_API_SECRET"
    dry_run: bool = True


class Settings(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    start_date: str
    end_date: str
    regime: RegimeSettings = Field(default_factory=RegimeSettings)
    strategy_breakout: StrategyBreakoutSettings = Field(default_factory=StrategyBreakoutSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    frictions: FrictionsSettings = Field(default_factory=FrictionsSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Settings.model_validate(data)
