from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationConfig:
    timeframe: str = "1h"
    horizon_candles: int = 3
    target_return: float = 0.006
    wick_threshold: float = 0.6
    sweep_lookback: int = 20
    min_support: int = 200
    regime_ma: int = 99
    train_window_candles: int = 24 * 30
    test_window_candles: int = 24 * 7
    slippage_bps: float = 2.0
    fee_bps: float = 4.0
    cooldown_candles: int = 0
    max_trades_per_day: int = 999
    enable_regime_adjustments: bool = True
    enable_regime_hard_block: bool = True
    regime_hard_block_id: str = "pa0_sl1_vh1"
    regime_rule_event: str = "break_structure_down"
    block_contrary_prob_gt: float = 0.60
    regime_flip_short_to_long: bool = True
    regime_long_size_mult: float = 1.25
    regime_short_size_mult: float = 1.0
    enable_payoff_filter: bool = True
    payoff_horizon: int = 3
    payoff_ratio_min: float = 1.20
    payoff_expected_min: float = 0.0
    payoff_models_path: str = ""
    payoff_train_split: float = 0.60
    payoff_fee_bps: float = 4.0
    payoff_slippage_bps: float = 2.0
    seed: int = 7
    output_dir: str = "reports"
