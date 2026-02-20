from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot.config import MarketStructureConfig
from trading_bot.strategies.market_structure_hhhl import pick_mode_a_target, structure_trailing_exit


@dataclass
class Position:
    side: int  # 1 long, -1 short
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    stop_price: float
    tp_price: float | None
    initial_r: float


def _risk_size(equity: float, entry: float, stop: float, cfg: MarketStructureConfig) -> float:
    risk_amount = equity * float(cfg.risk_per_trade)
    per_unit = abs(entry - stop)
    if per_unit <= 1e-12:
        return 0.0
    qty = risk_amount / per_unit
    max_notional = equity * float(cfg.leverage)
    qty_cap = max_notional / max(entry, 1e-12)
    return float(max(0.0, min(qty, qty_cap)))


def _apply_cost(side: int, price: float, is_entry: bool, cfg: MarketStructureConfig) -> float:
    slip = float(cfg.slippage_rate)
    if side > 0:
        return price * (1 + slip) if is_entry else price * (1 - slip)
    return price * (1 - slip) if is_entry else price * (1 + slip)


def _gross_pnl(side: int, entry: float, exit_p: float, qty: float) -> float:
    if side > 0:
        return (exit_p - entry) * qty
    return (entry - exit_p) * qty


def _evaluate_exit(position: Position, row: pd.Series, i: int, cfg: MarketStructureConfig) -> tuple[bool, str | None, float | None]:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])

    if position.side > 0 and low <= position.stop_price:
        return True, "stop", position.stop_price
    if position.side < 0 and high >= position.stop_price:
        return True, "stop", position.stop_price

    if position.tp_price is not None:
        if position.side > 0 and high >= position.tp_price:
            return True, "tp", position.tp_price
        if position.side < 0 and low <= position.tp_price:
            return True, "tp", position.tp_price

    if cfg.tp_mode == "C" and structure_trailing_exit(position.side, row):
        return True, "trailing_structure", close

    if cfg.time_stop_enabled and (i - position.entry_idx) >= int(cfg.time_stop_candles):
        return True, "time_stop", close

    return False, None, None


def run_backtest(df: pd.DataFrame, signals: pd.Series, cfg: MarketStructureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    fee = float(cfg.taker_fee_rate)
    equity = 10000.0
    positions: list[Position] = []
    trades: list[dict] = []
    curve: list[dict] = []

    max_open = 1 if cfg.one_position_only else max(1, int(cfg.max_positions))

    for i in range(len(df) - 1):
        row = df.iloc[i]
        ts = row["open_time"]

        to_close: list[tuple[Position, str, float]] = []
        for p in positions:
            close_now, reason, exit_raw = _evaluate_exit(p, row, i, cfg)
            if close_now and reason is not None and exit_raw is not None:
                to_close.append((p, reason, float(exit_raw)))

        for p, reason, exit_raw in to_close:
            exit_price = _apply_cost(p.side, float(exit_raw), is_entry=False, cfg=cfg)
            gross = _gross_pnl(p.side, p.entry_price, exit_price, p.qty)
            fees = (p.entry_price * p.qty + exit_price * p.qty) * fee
            pnl = gross - fees
            equity += pnl
            r_mult = 0.0 if p.initial_r <= 1e-12 else (_gross_pnl(p.side, p.entry_price, exit_price, 1.0) / p.initial_r)
            trades.append(
                {
                    "entry_time": p.entry_time,
                    "exit_time": ts,
                    "side": "LONG" if p.side > 0 else "SHORT",
                    "entry": p.entry_price,
                    "exit": exit_price,
                    "stop": p.stop_price,
                    "tp": p.tp_price,
                    "qty": p.qty,
                    "R": float(r_mult),
                    "pnl": pnl,
                    "exit_reason": reason,
                    "duration_candles": int(i - p.entry_idx),
                }
            )
            positions.remove(p)

        if len(positions) < max_open:
            side = int(signals.iat[i])
            if side != 0:
                entry_raw = float(row["close"])
                entry = _apply_cost(side, entry_raw, is_entry=True, cfg=cfg)

                if cfg.stop_mode == "atr":
                    atr = float(row.get("atr", np.nan))
                    if np.isnan(atr):
                        curve.append({"timestamp": ts, "equity": equity})
                        continue
                    stop = entry_raw - atr * cfg.atr_mult if side > 0 else entry_raw + atr * cfg.atr_mult
                else:
                    stop = float(row.get("last_swing_low_price", np.nan)) if side > 0 else float(row.get("last_swing_high_price", np.nan))
                    if np.isnan(stop):
                        curve.append({"timestamp": ts, "equity": equity})
                        continue

                qty = _risk_size(equity, entry, stop, cfg)
                if qty <= 0:
                    curve.append({"timestamp": ts, "equity": equity})
                    continue

                initial_r = abs(entry - stop)
                tp: float | None = None
                if cfg.tp_mode == "A":
                    tp = pick_mode_a_target(df, i, side, int(cfg.lookback_target))
                elif cfg.tp_mode == "B":
                    rr = float(cfg.rr_target)
                    tp = entry + (initial_r * rr) if side > 0 else entry - (initial_r * rr)

                positions.append(
                    Position(
                        side=side,
                        entry_idx=i,
                        entry_time=ts,
                        entry_price=entry,
                        qty=qty,
                        stop_price=stop,
                        tp_price=tp,
                        initial_r=initial_r,
                    )
                )

        curve.append({"timestamp": ts, "equity": equity})

    equity_df = pd.DataFrame(curve)
    trades_df = pd.DataFrame(trades)
    return trades_df, equity_df
