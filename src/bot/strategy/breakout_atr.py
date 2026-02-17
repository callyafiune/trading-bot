from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from bot.utils.config import StrategyBreakoutSettings

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency at runtime
    XGBClassifier = None


Side = Literal["LONG", "SHORT"]


@dataclass
class Signal:
    side: Side
    reason: str


@dataclass
class SignalDecision:
    signal: Signal | None
    blocked_reason: str | None
    raw_signal: Signal | None


class BreakoutATRStrategy:
    def __init__(self, settings: StrategyBreakoutSettings) -> None:
        self.settings = settings
        self.ml_probabilities = pd.Series(dtype=float)
        self.ml_threshold_by_index: dict[int, float] = {}
        self.selected_features: list[str] = []

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.settings.mode != "ml_gate":
            return out

        if XGBClassifier is None:
            raise RuntimeError("mode=ml_gate requer dependência xgboost instalada")

        feature_cols = [
            "log_return_1",
            "log_return_2",
            "log_return_4",
            "log_return_8",
            "ema_12",
            "ema_26",
            "macd_line",
            "rsi_14",
            "atr_pct",
            "realized_vol_24",
        ]
        model_df = out[feature_cols + ["close"]].copy()
        model_df["target"] = (model_df["close"].shift(-1) > model_df["close"]).astype(int)

        if not self.settings.use_walk_forward:
            out["ml_prob"] = np.nan
            out["ml_threshold"] = self.settings.ml_prob_threshold
            return out

        min_train = self.settings.wf_train_bars
        val_size = self.settings.wf_val_bars
        test_size = self.settings.wf_test_bars

        probs = pd.Series(np.nan, index=out.index, dtype=float)
        thresholds = pd.Series(self.settings.ml_prob_threshold, index=out.index, dtype=float)

        for start in range(min_train, len(out) - val_size - 1, test_size):
            train_idx = model_df.index[:start]
            val_idx = model_df.index[start : start + val_size]
            test_idx = model_df.index[start + val_size : start + val_size + test_size]
            if len(test_idx) == 0:
                continue

            train_xy = model_df.loc[train_idx].dropna()
            val_xy = model_df.loc[val_idx].dropna()
            test_xy = model_df.loc[test_idx].dropna()
            if train_xy.empty or val_xy.empty or test_xy.empty:
                continue

            selected = self._select_features(train_xy, feature_cols)
            self.selected_features = selected

            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )
            model.fit(train_xy[selected], train_xy["target"])

            val_prob = model.predict_proba(val_xy[selected])[:, 1]
            threshold = self._optimize_threshold(val_xy["target"].to_numpy(), val_prob)

            prob = model.predict_proba(test_xy[selected])[:, 1]
            probs.loc[test_xy.index] = prob
            thresholds.loc[test_xy.index] = threshold

        out["ml_prob"] = probs
        out["ml_threshold"] = thresholds
        return out

    def _select_features(self, train_xy: pd.DataFrame, feature_cols: list[str]) -> list[str]:
        # LightGBM-inspired selection by importance rank; fallback keeps all if selector unavailable.
        model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=7,
        )
        model.fit(train_xy[feature_cols], train_xy["target"])
        importances = model.feature_importances_
        ranking = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        top_k = max(2, min(self.settings.ml_feature_top_k, len(feature_cols)))
        return [name for name, _ in ranking[:top_k]]

    @staticmethod
    def _optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        best_t = 0.55
        best_score = -1.0
        for t in np.arange(0.50, 0.71, 0.01):
            preds = (y_prob >= t).astype(int)
            score = float((preds == y_true).mean())
            if score > best_score:
                best_score = score
                best_t = float(t)
        return best_t

    def evaluate_signal(self, df: pd.DataFrame, i: int) -> SignalDecision:
        row = df.iloc[i]
        raw_signal, raw_reason = self._raw_signal_by_mode(df, i)

        if row.get("regime") != "TREND":
            if raw_reason == "warmup":
                return SignalDecision(signal=None, blocked_reason="warmup", raw_signal=None)
            return SignalDecision(signal=None, blocked_reason="regime", raw_signal=raw_signal)

        if raw_signal is None:
            return SignalDecision(signal=None, blocked_reason=raw_reason, raw_signal=None)

        mode_block = self._mode_filter_reason(df, i, raw_signal)
        if mode_block:
            return SignalDecision(signal=None, blocked_reason=mode_block, raw_signal=raw_signal)

        ma_block = self._ma200_filter_reason(row, raw_signal)
        if ma_block:
            return SignalDecision(signal=None, blocked_reason=ma_block, raw_signal=raw_signal)

        return SignalDecision(signal=raw_signal, blocked_reason=None, raw_signal=raw_signal)

    def signal_decision(self, df: pd.DataFrame, i: int) -> tuple[Signal | None, str | None]:
        decision = self.evaluate_signal(df, i)
        return decision.signal, decision.blocked_reason

    def _raw_signal_by_mode(self, df: pd.DataFrame, i: int) -> tuple[Signal | None, str | None]:
        if self.settings.mode in ("breakout", "baseline"):
            if i < self.settings.breakout_lookback_N + 1:
                return None, "warmup"
            return self._breakout_signal(df, i)

        if i < 1:
            return None, "warmup"

        if self.settings.mode == "ema":
            return self._ema_signal(df, i)
        if self.settings.mode in ("ema_macd", "ml_gate"):
            return self._ema_signal(df, i)
        return None, "unsupported_mode"

    def _mode_filter_reason(self, df: pd.DataFrame, i: int, signal: Signal) -> str | None:
        direction = self.settings.trade_direction
        if direction == "long" and signal.side == "SHORT":
            return "direction"
        if direction == "short" and signal.side == "LONG":
            return "direction"

        row = df.iloc[i]
        if self.settings.mode in ("breakout", "baseline"):
            if self.settings.use_rel_volume_filter and row.get("rel_volume_24", 0.0) < self.settings.min_rel_volume:
                return "rel_volume"
            return None

        if self.settings.mode == "ema":
            return None

        if self.settings.mode in ("ema_macd", "ml_gate"):
            if self.settings.use_rel_volume_filter and row.get("rel_volume_24", 0.0) <= self.settings.min_rel_volume:
                return "rel_volume"

            macd_line = row.get("macd_line", np.nan)
            macd_signal = row.get("macd_signal", np.nan)
            if pd.isna(macd_line) or pd.isna(macd_signal):
                return "warmup"

            if signal.side == "LONG" and macd_line <= macd_signal:
                return "macd_gate"
            if signal.side == "SHORT" and macd_line >= macd_signal:
                return "macd_gate"

            if self.settings.mode == "ml_gate":
                prob = float(row.get("ml_prob", np.nan))
                threshold = float(row.get("ml_threshold", self.settings.ml_prob_threshold))
                if np.isnan(prob):
                    return "ml_warmup"
                if signal.side == "LONG" and prob < threshold:
                    return "ml_gate"
                if signal.side == "SHORT" and prob > (1.0 - threshold):
                    return "ml_gate"

        return None

    def _ma200_filter_reason(self, row: pd.Series, signal: Signal) -> str | None:
        if not self.settings.use_ma200_filter:
            return None

        ma_col = f"ma_{self.settings.ma200_period}"
        ma_value = row.get(ma_col, np.nan)
        close = row.get("close", np.nan)
        if pd.isna(ma_value) or pd.isna(close):
            return "ma200"

        if signal.side == "LONG" and close <= ma_value:
            return "ma200"
        if signal.side == "SHORT" and close >= ma_value:
            return "ma200"
        return None

    def _breakout_signal(self, df: pd.DataFrame, i: int) -> tuple[Signal | None, str | None]:
        row = df.iloc[i]
        lookback = df.iloc[i - self.settings.breakout_lookback_N : i]
        prev_high = lookback["high"].max()
        prev_low = lookback["low"].min()
        close = row["close"]

        if close > prev_high:
            return Signal(side="LONG", reason="breakout_high"), None
        if close < prev_low:
            return Signal(side="SHORT", reason="breakout_low"), None
        return None, "no_breakout"

    def _ema_signal(self, df: pd.DataFrame, i: int) -> tuple[Signal | None, str | None]:
        prev = df.iloc[i - 1]
        row = df.iloc[i]
        prev_fast, prev_slow = prev["ema_12"], prev["ema_26"]
        fast, slow = row["ema_12"], row["ema_26"]

        if any(pd.isna(v) for v in [prev_fast, prev_slow, fast, slow]):
            return None, "warmup"

        if prev_fast <= prev_slow and fast > slow:
            return Signal(side="LONG", reason="ema_cross_up"), None
        if prev_fast >= prev_slow and fast < slow:
            return Signal(side="SHORT", reason="ema_cross_down"), None
        return None, "no_crossover"

    def _ema_macd_signal(self, df: pd.DataFrame, i: int) -> tuple[Signal | None, str | None]:
        return self._ema_signal(df, i)

    def signal_at(self, df: pd.DataFrame, i: int) -> Signal | None:
        signal, _ = self.signal_decision(df, i)
        return signal

    def initial_stop(self, side: Side, entry_price: float, atr: float) -> float:
        k = self.settings.atr_k
        return entry_price - k * atr if side == "LONG" else entry_price + k * atr

    def trailing_stop(self, side: Side, curr_stop: float, close: float, atr: float) -> float:
        k = self.settings.atr_k
        candidate = close - k * atr if side == "LONG" else close + k * atr
        if side == "LONG":
            return max(curr_stop, candidate)
        return min(curr_stop, candidate)
