from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from backtests.baseline import generate_baseline_signals, prepare_pattern_frame
from backtests.config import ValidationConfig
from backtests.metrics import summarize_trades, write_json
from bot.pattern_mining.payoff_model import PayoffPredictor, predict_payoff


def _simulate_trade(
    frame: pd.DataFrame,
    i: int,
    side: int,
    horizon: int,
    target_return: float,
    slippage_rate: float,
    fee_rate: float,
    size_mult: float = 1.0,
) -> tuple[dict, int] | None:
    entry_raw = float(frame["close"].iat[i])
    entry_ts = frame["open_time"].iat[i]
    if pd.isna(entry_raw):
        return None

    if side > 0:
        entry_fill = entry_raw * (1.0 + slippage_rate)
        tp = entry_raw * (1.0 + target_return)
        sl = entry_raw * (1.0 - target_return)
    else:
        entry_fill = entry_raw * (1.0 - slippage_rate)
        tp = entry_raw * (1.0 - target_return)
        sl = entry_raw * (1.0 + target_return)

    exit_idx = i + horizon
    exit_reason = "horizon"
    exit_raw = float(frame["close"].iat[exit_idx])
    if pd.isna(exit_raw):
        return None

    for j in range(i + 1, min(i + horizon + 1, len(frame))):
        high = float(frame["high"].iat[j])
        low = float(frame["low"].iat[j])
        if side > 0:
            hit_tp = high >= tp
            hit_sl = low <= sl
            if hit_tp and hit_sl:
                exit_idx = j
                exit_raw = sl
                exit_reason = "sl_first"
                break
            if hit_sl:
                exit_idx = j
                exit_raw = sl
                exit_reason = "sl"
                break
            if hit_tp:
                exit_idx = j
                exit_raw = tp
                exit_reason = "tp"
                break
        else:
            hit_tp = low <= tp
            hit_sl = high >= sl
            if hit_tp and hit_sl:
                exit_idx = j
                exit_raw = sl
                exit_reason = "sl_first"
                break
            if hit_sl:
                exit_idx = j
                exit_raw = sl
                exit_reason = "sl"
                break
            if hit_tp:
                exit_idx = j
                exit_raw = tp
                exit_reason = "tp"
                break

    if side > 0:
        exit_fill = exit_raw * (1.0 - slippage_rate)
        gross = (exit_fill / entry_fill) - 1.0
    else:
        exit_fill = exit_raw * (1.0 + slippage_rate)
        gross = (entry_fill / exit_fill) - 1.0

    net = (gross - (2.0 * fee_rate)) * float(size_mult)
    trade = {
        "entry_time": entry_ts,
        "exit_time": frame["open_time"].iat[exit_idx],
        "side": "LONG" if side > 0 else "SHORT",
        "entry_price": entry_fill,
        "exit_price": exit_fill,
        "raw_entry_price": entry_raw,
        "raw_exit_price": exit_raw,
        "pnl": net,
        "return": net,
        "hold_candles": int(exit_idx - i),
        "exit_reason": exit_reason,
    }
    return trade, exit_idx


def run_execution_simulation(
    df: pd.DataFrame,
    cfg: ValidationConfig,
    prepared_df: pd.DataFrame | None = None,
    signals: pd.Series | None = None,
    payoff_predictor: PayoffPredictor | None = None,
    persist: bool = True,
) -> dict:
    frame = prepared_df.copy() if prepared_df is not None else prepare_pattern_frame(df, cfg)
    sig = signals.copy() if signals is not None else generate_baseline_signals(frame, cfg)

    slippage_rate = float(cfg.slippage_bps) / 10000.0
    fee_rate = float(cfg.fee_bps) / 10000.0
    horizon = int(cfg.horizon_candles)

    trades: list[dict] = []
    trades_per_day: dict[str, int] = defaultdict(int)
    cooldown_until = -1
    blocked_by_regime_rule_total = 0
    blocked_edge_total = 0
    adjusted_signal_total = 0
    skipped_invalid_trade_total = 0
    blocked_by_payoff_total = 0

    i = 0
    while i < len(frame) - horizon:
        if i <= cooldown_until:
            i += 1
            continue

        side = int(sig.iat[i])
        if side == 0:
            i += 1
            continue

        pred_runup = None
        pred_ddown_abs = None
        payoff_expected = None
        payoff_ratio = None
        if cfg.enable_payoff_filter:
            if "pred_runup" in frame.columns and "pred_ddown_abs" in frame.columns:
                val_runup = frame["pred_runup"].iat[i]
                val_ddown = frame["pred_ddown_abs"].iat[i]
                if pd.notna(val_runup) and pd.notna(val_ddown):
                    pred_runup = float(val_runup)
                    pred_ddown_abs = float(max(0.0, val_ddown))
            if pred_runup is None or pred_ddown_abs is None:
                row_payload = frame.iloc[i].to_dict()
                pred_runup, pred_ddown_abs = predict_payoff(row_payload, payoff_predictor)
            if pred_runup is not None and pred_ddown_abs is not None:
                cost_fee = float(cfg.payoff_fee_bps or cfg.fee_bps) / 10000.0
                cost_slip = float(cfg.payoff_slippage_bps or cfg.slippage_bps) / 10000.0
                costs = cost_fee + cost_slip
                payoff_expected = float(pred_runup - pred_ddown_abs - costs)
                payoff_ratio = float(pred_runup / max(pred_ddown_abs, 1e-6))
                if payoff_expected < float(cfg.payoff_expected_min) or payoff_ratio < float(cfg.payoff_ratio_min):
                    blocked_by_payoff_total += 1
                    i += 1
                    continue

        event_col = cfg.regime_rule_event
        event_flag = bool(frame.get(event_col, pd.Series(False, index=frame.index)).iat[i])
        regime_id = str(frame.get("regime_id", pd.Series("", index=frame.index)).iat[i])
        regime_match = event_flag and regime_id == cfg.regime_hard_block_id

        if cfg.enable_regime_adjustments and cfg.enable_regime_hard_block and side < 0:
            if regime_match:
                blocked_by_regime_rule_total += 1
                if cfg.regime_flip_short_to_long:
                    side = 1
                    adjusted_signal_total += 1
                else:
                    i += 1
                    continue

        day_key = str(frame["open_time"].iat[i].date())
        if trades_per_day[day_key] >= int(cfg.max_trades_per_day):
            i += 1
            continue

        size_mult = 1.0
        if cfg.enable_regime_adjustments and regime_match:
            if side > 0:
                size_mult = float(cfg.regime_long_size_mult)
            else:
                size_mult = float(cfg.regime_short_size_mult)

        trade_result = _simulate_trade(
            frame,
            i,
            side,
            horizon,
            cfg.target_return,
            slippage_rate,
            fee_rate,
            size_mult=size_mult,
        )
        if trade_result is None:
            skipped_invalid_trade_total += 1
            i += 1
            continue
        trade, exit_idx = trade_result
        trade["pred_runup"] = pred_runup
        trade["pred_ddown_abs"] = pred_ddown_abs
        trade["payoff_expected"] = payoff_expected
        trade["payoff_ratio"] = payoff_ratio

        trades.append(trade)
        trades_per_day[day_key] += 1
        cooldown_until = exit_idx + int(cfg.cooldown_candles)
        i = exit_idx + 1

    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades_df)
    summary["avg_hold"] = float(trades_df["hold_candles"].mean()) if not trades_df.empty else 0.0
    summary["blocked_edge_total"] = int(blocked_edge_total)
    summary["blocked_by_regime_rule_total"] = int(blocked_by_regime_rule_total)
    summary["adjusted_signal_total"] = int(adjusted_signal_total)
    summary["skipped_invalid_trade_total"] = int(skipped_invalid_trade_total)
    summary["blocked_by_payoff_total"] = int(blocked_by_payoff_total)

    if trades_df.empty:
        equity_df = pd.DataFrame(columns=["timestamp", "equity"])
    else:
        eq = 1.0
        curve = [{"timestamp": frame["open_time"].iat[0], "equity": eq}]
        for _, row in trades_df.iterrows():
            eq *= 1.0 + float(row["pnl"])
            curve.append({"timestamp": row["exit_time"], "equity": eq})
        equity_df = pd.DataFrame(curve)

    if persist:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "execution_metrics.json", summary)
        trades_df.to_csv(out_dir / "execution_trades.csv", index=False)
        equity_df.to_csv(out_dir / "equity_curve.csv", index=False)

    return {
        "metrics": summary,
        "trades": trades_df,
        "equity_curve": equity_df,
    }
