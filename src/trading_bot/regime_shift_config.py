from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

SignificanceMode = Literal["atr", "percent"]
BreakConfirmMode = Literal["close_only", "close_plus_buffer", "n_closes"]
BufferMode = Literal["atr", "percent"]
StopMode = Literal["major_level", "atr"]
TPMode = Literal["rr", "trailing_major"]


@dataclass
class RegimeShiftConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    start_date: str = "2023-01-01"
    end_date: str = "2026-12-31"
    timezone: str = "UTC"

    lookback_init: int = 48
    significance_mode: SignificanceMode = "atr"
    atr_len: int = 14
    atr_k: float = 2.0
    percent_p: float = 0.008

    break_confirm_mode: BreakConfirmMode = "close_plus_buffer"
    buffer_mode: BufferMode = "atr"
    atr_buffer_k: float = 0.25
    percent_buffer_p: float = 0.001
    n_closes_confirm: int = 2

    allow_flip: bool = True
    one_position_only: bool = True

    stop_mode: StopMode = "major_level"
    stop_atr_k: float = 2.0

    tp_mode: TPMode = "rr"
    rr_target: float = 2.0

    taker_fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    risk_per_trade: float = 0.01
    leverage: float = 2.0

    filter_ma99: bool = False

    seed: int = 7
    output_dir: str = "runs/regime_shift"
    cache_dir: str = "data/cache"

    grid_lookback_init: list[int] | None = None
    grid_atr_k: list[float] | None = None
    grid_percent_p: list[float] | None = None
    grid_break_confirm_mode: list[str] | None = None
    grid_atr_buffer_k: list[float] | None = None
    grid_percent_buffer_p: list[float] | None = None
    grid_stop_atr_k: list[float] | None = None
    grid_rr_target: list[float] | None = None
    grid_allow_flip: list[bool] | None = None
    grid_filter_ma99: list[bool] | None = None
    grid_max_combinations: int = 300


def load_regime_shift_config(path: str | Path) -> RegimeShiftConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = RegimeShiftConfig()
    for key, val in payload.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


def build_grid_configs(base: RegimeShiftConfig) -> list[RegimeShiftConfig]:
    lookbacks = base.grid_lookback_init or [24, 48, 72]
    atr_ks = base.grid_atr_k or [1.5, 2.0, 2.5]
    pct_ps = base.grid_percent_p or [0.005, 0.008, 0.01]
    break_modes = base.grid_break_confirm_mode or ["close_only", "close_plus_buffer", "n_closes"]
    atr_bufs = base.grid_atr_buffer_k or [0.0, 0.25, 0.5]
    pct_bufs = base.grid_percent_buffer_p or [0.0, 0.001, 0.002]
    stop_atrs = base.grid_stop_atr_k or [2.0, 3.0]
    rr_targets = base.grid_rr_target or [1.5, 2.0, 3.0]
    flips = base.grid_allow_flip or [True, False]
    ma99s = base.grid_filter_ma99 or [False, True]

    out: list[RegimeShiftConfig] = []
    for lb in lookbacks:
        for sig_mode in ("atr", "percent"):
            sig_values = atr_ks if sig_mode == "atr" else pct_ps
            for sig_v in sig_values:
                for bmode in break_modes:
                    for buf_mode in ("atr", "percent"):
                        buf_values = atr_bufs if buf_mode == "atr" else pct_bufs
                        for buf_v in buf_values:
                            for stop_mode in ("major_level", "atr"):
                                stop_values = stop_atrs if stop_mode == "atr" else [base.stop_atr_k]
                                for stop_v in stop_values:
                                    for tp_mode in ("rr", "trailing_major"):
                                        rr_vals = rr_targets if tp_mode == "rr" else [base.rr_target]
                                        for rr in rr_vals:
                                            for fl in flips:
                                                for ma in ma99s:
                                                    c = RegimeShiftConfig(**base.__dict__)
                                                    c.lookback_init = int(lb)
                                                    c.significance_mode = sig_mode  # type: ignore[assignment]
                                                    if sig_mode == "atr":
                                                        c.atr_k = float(sig_v)
                                                    else:
                                                        c.percent_p = float(sig_v)
                                                    c.break_confirm_mode = bmode  # type: ignore[assignment]
                                                    c.buffer_mode = buf_mode  # type: ignore[assignment]
                                                    if buf_mode == "atr":
                                                        c.atr_buffer_k = float(buf_v)
                                                    else:
                                                        c.percent_buffer_p = float(buf_v)
                                                    c.stop_mode = stop_mode  # type: ignore[assignment]
                                                    c.stop_atr_k = float(stop_v)
                                                    c.tp_mode = tp_mode  # type: ignore[assignment]
                                                    c.rr_target = float(rr)
                                                    c.allow_flip = bool(fl)
                                                    c.filter_ma99 = bool(ma)
                                                    out.append(c)

    if len(out) <= int(base.grid_max_combinations):
        return out

    import random

    rnd = random.Random(int(base.seed))
    idx = list(range(len(out)))
    rnd.shuffle(idx)
    idx = sorted(idx[: int(base.grid_max_combinations)])
    return [out[i] for i in idx]
