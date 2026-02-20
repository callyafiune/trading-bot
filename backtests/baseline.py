from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtests.config import ValidationConfig
from backtests.metrics import summarize_trades, write_json
from bot.pattern_mining.dataset_builder import build_synchronized_dataset
from bot.pattern_mining.event_detection import detect_events
from bot.pattern_mining.feature_engineering import build_pattern_features, build_regime_flags
from bot.pattern_mining.labeling import add_labels


def prepare_pattern_frame(df: pd.DataFrame, cfg: ValidationConfig) -> pd.DataFrame:
    base = build_synchronized_dataset(df)
    feat = build_pattern_features(base)
    events = detect_events(
        feat,
        big_lower_wick_threshold=cfg.wick_threshold,
        sweep_lookback=cfg.sweep_lookback,
        structure_lookback=cfg.sweep_lookback,
    )
    events = build_regime_flags(events)
    labeled = add_labels(
        events,
        horizon=cfg.horizon_candles,
        up_threshold=cfg.target_return,
        down_threshold=-cfg.target_return,
    )
    return labeled


def generate_baseline_signals(frame: pd.DataFrame, cfg: ValidationConfig) -> pd.Series:
    sig = pd.Series(0, index=frame.index, dtype=int)

    long_sweep = frame.get("sweep_down_reclaim", False).fillna(False)
    short_msb = frame.get("break_structure_down", False).fillna(False)

    if "oi" in frame.columns and "var_oi_1" in frame.columns and frame["oi"].notna().any():
        long_wick = frame.get("big_lower_wick", False).fillna(False) & (frame["var_oi_1"] < 0)
    else:
        long_wick = frame.get("big_lower_wick", False).fillna(False)

    sig[long_sweep | long_wick] = 1
    sig[short_msb] = -1
    sig[(long_sweep | long_wick) & short_msb] = 0
    return sig


def _simulate_close_to_close(frame: pd.DataFrame, signals: pd.Series, cfg: ValidationConfig) -> pd.DataFrame:
    fee = float(cfg.fee_bps) / 10000.0
    horizon = int(cfg.horizon_candles)

    trades: list[dict] = []
    for i in range(len(frame) - horizon):
        side = int(signals.iat[i])
        if side == 0:
            continue
        entry = float(frame["close"].iat[i])
        exit_price = float(frame["close"].iat[i + horizon])
        gross = (exit_price / entry - 1.0) if side > 0 else (entry / exit_price - 1.0)
        net = gross - (2.0 * fee)
        trades.append(
            {
                "entry_time": frame["open_time"].iat[i],
                "exit_time": frame["open_time"].iat[i + horizon],
                "side": "LONG" if side > 0 else "SHORT",
                "entry_price": entry,
                "exit_price": exit_price,
                "return": net,
                "pnl": net,
                "hold_candles": horizon,
            }
        )
    return pd.DataFrame(trades)


def run_baseline(
    df: pd.DataFrame,
    cfg: ValidationConfig,
    prepared_df: pd.DataFrame | None = None,
    signals: pd.Series | None = None,
    persist: bool = True,
) -> dict:
    frame = prepared_df.copy() if prepared_df is not None else prepare_pattern_frame(df, cfg)
    sig = signals.copy() if signals is not None else generate_baseline_signals(frame, cfg)

    trades = _simulate_close_to_close(frame, sig, cfg)
    metrics = summarize_trades(trades)

    has_label = frame[f"future_return_{cfg.horizon_candles}h"].notna()
    action = sig != 0
    mask = has_label & action
    correct = ((sig == 1) & frame["y_up"]) | ((sig == -1) & frame["y_down"])

    label_metrics = {
        "signal_count": int(mask.sum()),
        "directional_accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
        "long_accuracy_y_up": float(frame.loc[sig == 1, "y_up"].mean()) if (sig == 1).any() else 0.0,
        "short_accuracy_y_down": float(frame.loc[sig == -1, "y_down"].mean()) if (sig == -1).any() else 0.0,
    }

    result = {
        "metrics": metrics,
        "label_metrics": label_metrics,
        "trades": trades,
        "signals": sig,
        "frame": frame,
    }

    if persist:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "baseline_metrics.json", metrics)
        write_json(out_dir / "baseline_label_metrics.json", label_metrics)
        trades.to_csv(out_dir / "baseline_trades.csv", index=False)

    return result
