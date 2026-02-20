from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot.regime_shift_config import RegimeShiftConfig


@dataclass
class Position:
    side: int
    entry_idx: int
    entry_time: pd.Timestamp
    entry: float
    qty: float
    stop: float
    tp: float | None
    initial_r: float


def _size(equity: float, entry: float, stop: float, cfg: RegimeShiftConfig) -> float:
    risk_amt = equity * float(cfg.risk_per_trade)
    dist = abs(entry - stop)
    if dist <= 1e-12:
        return 0.0
    qty = risk_amt / dist
    cap = (equity * float(cfg.leverage)) / max(entry, 1e-12)
    return float(max(0.0, min(qty, cap)))


def _exec_price(side: int, price: float, is_entry: bool, cfg: RegimeShiftConfig) -> float:
    slip = float(cfg.slippage_rate)
    if side > 0:
        return price * (1 + slip) if is_entry else price * (1 - slip)
    return price * (1 - slip) if is_entry else price * (1 + slip)


def _pnl(side: int, entry: float, exit_p: float, qty: float) -> float:
    return (exit_p - entry) * qty if side > 0 else (entry - exit_p) * qty


def run_regime_shift_backtest(df: pd.DataFrame, cfg: RegimeShiftConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    equity = 10000.0
    fee = float(cfg.taker_fee_rate)
    pos: Position | None = None
    trades: list[dict] = []
    curve: list[dict] = []

    for i in range(len(df)):
        row = df.iloc[i]
        ts = pd.Timestamp(row["open_time"])
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row.get("atr", np.nan))

        bull_shift = bool(row.get("bullish_shift", False))
        bear_shift = bool(row.get("bearish_shift", False))

        if cfg.filter_ma99:
            ma99 = float(row.get("ma99", np.nan))
            if np.isnan(ma99):
                bull_shift = False
                bear_shift = False
            else:
                if close <= ma99:
                    bull_shift = False
                if close >= ma99:
                    bear_shift = False

        if pos is not None:
            if cfg.stop_mode == "major_level":
                if pos.side > 0:
                    ml = float(row.get("major_low", np.nan))
                    if not np.isnan(ml):
                        pos.stop = max(pos.stop, ml)
                else:
                    mh = float(row.get("major_high", np.nan))
                    if not np.isnan(mh):
                        pos.stop = min(pos.stop, mh)

            reason = None
            exit_raw = None
            # priority: stop > tp > flip
            if pos.side > 0 and low <= pos.stop:
                reason = "stop"
                exit_raw = pos.stop
            elif pos.side < 0 and high >= pos.stop:
                reason = "stop"
                exit_raw = pos.stop
            elif pos.tp is not None:
                if pos.side > 0 and high >= pos.tp:
                    reason = "tp"
                    exit_raw = pos.tp
                elif pos.side < 0 and low <= pos.tp:
                    reason = "tp"
                    exit_raw = pos.tp

            flip_to = None
            if reason is None:
                if pos.side > 0 and bear_shift:
                    reason = "flip"
                    exit_raw = close
                    flip_to = -1
                elif pos.side < 0 and bull_shift:
                    reason = "flip"
                    exit_raw = close
                    flip_to = 1

            if reason is not None and exit_raw is not None:
                exit_exec = _exec_price(pos.side, float(exit_raw), is_entry=False, cfg=cfg)
                gross = _pnl(pos.side, pos.entry, exit_exec, pos.qty)
                fees = (pos.entry * pos.qty + exit_exec * pos.qty) * fee
                pnl = gross - fees
                equity += pnl
                r_val = 0.0 if pos.initial_r <= 1e-12 else (_pnl(pos.side, pos.entry, exit_exec, 1.0) / pos.initial_r)
                trades.append(
                    {
                        "entry_time": pos.entry_time,
                        "exit_time": ts,
                        "side": "LONG" if pos.side > 0 else "SHORT",
                        "entry": pos.entry,
                        "exit": exit_exec,
                        "stop": pos.stop,
                        "tp": pos.tp,
                        "R": float(r_val),
                        "qty": pos.qty,
                        "pnl": float(pnl),
                        "reason": reason,
                        "duration_candles": int(i - pos.entry_idx),
                    }
                )
                pos = None

                if flip_to is not None and cfg.allow_flip:
                    side = int(flip_to)
                    entry_raw = close
                    entry_exec = _exec_price(side, entry_raw, is_entry=True, cfg=cfg)
                    if cfg.stop_mode == "major_level":
                        stop = float(row.get("major_low", np.nan)) if side > 0 else float(row.get("major_high", np.nan))
                    else:
                        if np.isnan(atr):
                            stop = np.nan
                        else:
                            stop = entry_raw - atr * float(cfg.stop_atr_k) if side > 0 else entry_raw + atr * float(cfg.stop_atr_k)
                    if not np.isnan(stop):
                        qty = _size(equity, entry_exec, stop, cfg)
                        if qty > 0:
                            init_r = abs(entry_exec - stop)
                            tp = entry_exec + init_r * cfg.rr_target if (cfg.tp_mode == "rr" and side > 0) else None
                            if cfg.tp_mode == "rr" and side < 0:
                                tp = entry_exec - init_r * cfg.rr_target
                            pos = Position(side, i, ts, entry_exec, qty, stop, tp, init_r)

        if pos is None:
            side = 1 if bull_shift else (-1 if bear_shift else 0)
            if side != 0:
                entry_raw = close
                entry_exec = _exec_price(side, entry_raw, is_entry=True, cfg=cfg)

                if cfg.stop_mode == "major_level":
                    stop = float(row.get("major_low", np.nan)) if side > 0 else float(row.get("major_high", np.nan))
                else:
                    if np.isnan(atr):
                        stop = np.nan
                    else:
                        stop = entry_raw - atr * float(cfg.stop_atr_k) if side > 0 else entry_raw + atr * float(cfg.stop_atr_k)

                if not np.isnan(stop):
                    qty = _size(equity, entry_exec, stop, cfg)
                    if qty > 0:
                        init_r = abs(entry_exec - stop)
                        tp = None
                        if cfg.tp_mode == "rr":
                            tp = entry_exec + init_r * cfg.rr_target if side > 0 else entry_exec - init_r * cfg.rr_target
                        pos = Position(side, i, ts, entry_exec, qty, stop, tp, init_r)

        curve.append({"timestamp": ts, "equity": equity})

    return pd.DataFrame(trades), pd.DataFrame(curve)
