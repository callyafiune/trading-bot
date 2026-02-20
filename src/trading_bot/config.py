from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


TPMode = Literal["A", "B", "C"]
StopMode = Literal["swing", "atr"]


@dataclass
class MarketStructureConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    start_date: str = "2023-01-01"
    end_date: str = "2026-12-31"
    timezone: str = "UTC"

    pivot_left: int = 2
    pivot_right: int = 2
    use_close_for_swings: bool = False
    enter_on_breakout: bool = False

    tp_mode: TPMode = "A"
    rr_target: float = 2.0
    lookback_target: int = 300

    stop_mode: StopMode = "swing"
    atr_length: int = 14
    atr_mult: float = 2.0

    taker_fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    risk_per_trade: float = 0.01
    leverage: float = 2.0

    one_position_only: bool = True
    max_positions: int = 1

    filter_ma99: bool = False
    filter_trend_ma: bool = False
    trend_ma_length: int = 200
    filter_atr_vol: bool = False
    atr_pct_min: float = 0.0
    filter_session: bool = False
    allowed_hours_utc: list[int] | None = None

    time_stop_enabled: bool = False
    time_stop_candles: int = 72

    seed: int = 7
    output_dir: str = "runs/market_structure"
    cache_dir: str = "data/cache"

    grid_pivots: list[tuple[int, int]] | None = None
    grid_tp_mode: list[str] | None = None
    grid_rr_values: list[float] | None = None
    grid_stop_mode: list[str] | None = None
    grid_atr_mult: list[float] | None = None
    grid_filter_ma99: list[bool] | None = None


def load_config(path: str | Path) -> MarketStructureConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = MarketStructureConfig()
    for key, val in payload.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


def build_grid_configs(base: MarketStructureConfig) -> list[MarketStructureConfig]:
    pivots = base.grid_pivots or [(2, 2), (3, 3), (4, 4)]
    tp_modes = base.grid_tp_mode or ["A", "B"]
    rr_values = base.grid_rr_values or [1.5, 2.0, 3.0]
    stop_modes = base.grid_stop_mode or ["swing", "atr"]
    atr_mults = base.grid_atr_mult or [2.0, 3.0]
    ma99_vals = base.grid_filter_ma99 or [False, True]

    configs: list[MarketStructureConfig] = []
    for pl, pr in pivots:
        for tp_mode in tp_modes:
            for stop_mode in stop_modes:
                for ma99 in ma99_vals:
                    rr_iter = rr_values if tp_mode == "B" else [base.rr_target]
                    atr_iter = atr_mults if stop_mode == "atr" else [base.atr_mult]
                    for rr in rr_iter:
                        for atr_mult in atr_iter:
                            c = MarketStructureConfig(**base.__dict__)
                            c.pivot_left = int(pl)
                            c.pivot_right = int(pr)
                            c.tp_mode = tp_mode  # type: ignore[assignment]
                            c.stop_mode = stop_mode  # type: ignore[assignment]
                            c.rr_target = float(rr)
                            c.atr_mult = float(atr_mult)
                            c.filter_ma99 = bool(ma99)
                            configs.append(c)
    return configs
