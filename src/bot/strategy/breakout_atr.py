from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from bot.market_structure.detector import add_market_structure_features
from bot.utils.config import (
    FngFilterSettings,
    FundingFilterSettings,
    MarketStructureSettings,
    MultiTimeframeSettings,
    RouterSettings,
    StrategyBreakoutSettings,
    StrategyRouterSettings,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency at runtime
    XGBClassifier = None


Side = Literal["LONG", "SHORT"]


@dataclass
class Signal:
    side: Side
    reason: str
    entry_type: str = "direct"


@dataclass
class SignalDecision:
    signal: Signal | None
    blocked_reason: str | None
    raw_signal: Signal | None


class BreakoutATRStrategy:
    def __init__(
        self,
        settings: StrategyBreakoutSettings,
        router_settings: StrategyRouterSettings | None = None,
        router_policy: RouterSettings | None = None,
        funding_filter: FundingFilterSettings | None = None,
        fng_filter: FngFilterSettings | None = None,
        mtf_settings: MultiTimeframeSettings | None = None,
        market_structure: MarketStructureSettings | None = None,
    ) -> None:
        self.settings = settings
        self.router_settings = router_settings or StrategyRouterSettings()
        self.router_policy = router_policy or RouterSettings()
        self.funding_filter = funding_filter or FundingFilterSettings()
        self.fng_filter = fng_filter or FngFilterSettings()
        self.mtf_settings = mtf_settings or MultiTimeframeSettings()
        self.market_structure = market_structure or MarketStructureSettings()
        self.ml_probabilities = pd.Series(dtype=float)
        self.ml_threshold_by_index: dict[int, float] = {}
        self.selected_features: list[str] = []
        self.pending_retest: dict[str, dict[str, float | int]] = {}

    def _attach_funding_features(self, out: pd.DataFrame) -> pd.DataFrame:
        out = out.copy()
        out["funding_action"] = "none"
        if "funding_rate" not in out.columns:
            return out

        window = max(2, int(self.funding_filter.z_window))
        mean = out["funding_rate"].rolling(window, min_periods=window).mean()
        std = out["funding_rate"].rolling(window, min_periods=window).std(ddof=0).replace(0.0, np.nan)
        out["funding_z"] = (out["funding_rate"] - mean) / std
        if not self.funding_filter.enabled:
            return out

        z = out["funding_z"]
        long_thr = float(self.funding_filter.block_long_if_z_gt)
        short_thr = float(self.funding_filter.block_short_if_z_lt)
        z_abs = abs(float(self.funding_filter.z_threshold))
        long_block = z > max(long_thr, z_abs)
        short_block = z < min(short_thr, -z_abs)
        out.loc[long_block, "funding_action"] = "block_long"
        out.loc[short_block, "funding_action"] = "block_short"
        return out

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        self.pending_retest = {}
        out = self._attach_funding_features(df.copy())
        if self.market_structure.enabled:
            required = {"ms_structure_state", "msb_bull", "msb_bear"}
            if not required.issubset(set(out.columns)):
                out = add_market_structure_features(out, self.market_structure)
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
        mode = self._mode_for_row(row)
        raw_signal, raw_reason = self._raw_signal_by_mode(df, i, mode)

        if raw_signal is None:
            return SignalDecision(signal=None, blocked_reason=raw_reason, raw_signal=None)

        regime_block = self._router_block_reason(row, raw_signal, mode)
        if regime_block:
            return SignalDecision(signal=None, blocked_reason=regime_block, raw_signal=raw_signal)

        funding_block = self._funding_filter_reason(row, raw_signal)
        if funding_block:
            return SignalDecision(signal=None, blocked_reason=funding_block, raw_signal=raw_signal)

        fng_block = self._fng_filter_reason(row, raw_signal)
        if fng_block:
            return SignalDecision(signal=None, blocked_reason=fng_block, raw_signal=raw_signal)

        mode_block = self._mode_filter_reason(df, i, raw_signal, mode)
        if mode_block:
            return SignalDecision(signal=None, blocked_reason=mode_block, raw_signal=raw_signal)

        ma_block = self._ma200_filter_reason(row, raw_signal)
        if ma_block:
            return SignalDecision(signal=None, blocked_reason=ma_block, raw_signal=raw_signal)

        mtf_block = self._mtf_filter_reason(row, raw_signal)
        if mtf_block:
            return SignalDecision(signal=None, blocked_reason=mtf_block, raw_signal=raw_signal)

        ms_block = self._market_structure_gate_reason(row, raw_signal)
        if ms_block:
            return SignalDecision(signal=None, blocked_reason=ms_block, raw_signal=raw_signal)

        return SignalDecision(signal=raw_signal, blocked_reason=None, raw_signal=raw_signal)

    def _market_structure_gate_reason(self, row: pd.Series, signal: Signal) -> str | None:
        if not (self.market_structure.enabled and self.market_structure.gate.enabled):
            return None

        structure_state = str(row.get("ms_structure_state", "NEUTRAL"))
        bull_structure = structure_state == "BULLISH"
        bear_structure = structure_state == "BEARISH"
        bull_msb = bool(row.get("msb_bull_active", row.get("msb_bull", False)))
        bear_msb = bool(row.get("msb_bear_active", row.get("msb_bear", False)))

        def token_active(token: str) -> bool:
            if token == "BULLISH_MSB":
                return bull_msb
            if token == "BEARISH_MSB":
                return bear_msb
            if token == "BULLISH_STRUCTURE":
                return bull_structure
            if token == "BEARISH_STRUCTURE":
                return bear_structure
            return False

        if signal.side == "LONG":
            allow_tokens = list(self.market_structure.gate.allow_long_when)
            mode_ok = self._market_structure_mode_allows(side="LONG", bull_structure=bull_structure, bear_structure=bear_structure, bull_msb=bull_msb, bear_msb=bear_msb)
            token_ok = any(token_active(token) for token in allow_tokens) if allow_tokens else mode_ok
            if self.market_structure.gate.block_in_neutral and structure_state == "NEUTRAL" and not bull_msb:
                return "ms_gate_long"
            return None if (mode_ok and token_ok) else "ms_gate_long"

        allow_tokens = list(self.market_structure.gate.allow_short_when)
        mode_ok = self._market_structure_mode_allows(side="SHORT", bull_structure=bull_structure, bear_structure=bear_structure, bull_msb=bull_msb, bear_msb=bear_msb)
        token_ok = any(token_active(token) for token in allow_tokens) if allow_tokens else mode_ok
        if self.market_structure.gate.block_in_neutral and structure_state == "NEUTRAL" and not bear_msb:
            return "ms_gate_short"
        return None if (mode_ok and token_ok) else "ms_gate_short"

    def _market_structure_mode_allows(
        self,
        *,
        side: Side,
        bull_structure: bool,
        bear_structure: bool,
        bull_msb: bool,
        bear_msb: bool,
    ) -> bool:
        mode = self.market_structure.gate.mode
        if side == "LONG":
            if mode == "msb_only":
                return bull_msb
            if mode == "structure_trend":
                return bull_structure
            if self.market_structure.gate.hybrid_require_both:
                return bull_structure and bull_msb
            return bull_structure or bull_msb

        if mode == "msb_only":
            return bear_msb
        if mode == "structure_trend":
            return bear_structure
        if self.market_structure.gate.hybrid_require_both:
            return bear_structure and bear_msb
        return bear_structure or bear_msb

    def signal_decision(self, df: pd.DataFrame, i: int) -> tuple[Signal | None, str | None]:
        decision = self.evaluate_signal(df, i)
        return decision.signal, decision.blocked_reason

    def _mode_for_row(self, row: pd.Series) -> str:
        if not self.router_policy.enabled:
            return self.settings.mode
        regime = str(row.get("regime", ""))
        micro = regime.split("_")[-1] if "_" in regime else regime
        if micro == "RANGE":
            return self.router_policy.range_mode
        return self.router_policy.trend_mode

    def _router_block_reason(self, row: pd.Series, signal: Signal, mode: str) -> str | None:
        regime = str(row.get("regime", ""))
        macro = regime.split("_")[0] if "_" in regime else regime
        micro = regime.split("_")[-1] if "_" in regime else regime

        if regime == "TREND_UP":
            macro, micro = "BULL", "TREND"
        elif regime == "TREND_DOWN":
            macro, micro = "BEAR", "TREND"
        elif regime == "RANGE":
            macro, micro = "TRANSITION", "RANGE"
        elif regime == "CHAOS":
            macro, micro = "TRANSITION", "CHAOS"

        if micro == "CHAOS":
            if self.router_policy.enabled:
                return None if self.router_policy.chaos_trade else "blocked_chaos_flat"
            return None if self.router_settings.enable_chaos else "blocked_chaos_flat"

        if micro == "RANGE":
            if not self.router_settings.enable_range:
                return "blocked_range_flat"

        if self.settings.trade_direction == "long":
            return "direction" if signal.side == "SHORT" else None
        if self.settings.trade_direction == "short":
            return "direction" if signal.side == "LONG" else None

        if macro == "BULL":
            if not self.router_settings.enable_trend_up_long:
                return "blocked_trend_up_flat"
            if signal.side == "SHORT":
                return "blocked_trend_up_short_only"

            if mode not in ("breakout", "baseline"):
                return None

            close = row.get("close", np.nan)
            ema200 = row.get("ema200", np.nan)
            ema50 = row.get("ema50", np.nan)
            slope_ema200_pct = row.get("slope_ema200_pct", row.get("slope_ema200", np.nan))
            if pd.isna(close) or pd.isna(ema200) or pd.isna(ema50) or pd.isna(slope_ema200_pct):
                return "blocked_trend_up_filter"
            if close <= ema200 or ema50 <= ema200 or slope_ema200_pct < self.router_settings.bull_slope_min:
                return "blocked_trend_up_filter"
            return None
        if macro == "BEAR":
            if not self.router_settings.enable_trend_down_short:
                return "blocked_trend_down_flat"
            return "blocked_trend_down_long_only" if signal.side == "LONG" else None
        if micro == "TREND":
            return None
        return "blocked_regime_unknown"

    def _breakout_lookback_for_row(self, row: pd.Series) -> int:
        regime = str(row.get("regime", ""))
        macro = regime.split("_")[0] if "_" in regime else regime
        if regime == "TREND_UP":
            macro = "BULL"
        elif regime == "TREND_DOWN":
            macro = "BEAR"

        if macro == "BULL":
            return max(1, int(self.router_settings.overrides.bull_trend.breakout_N))
        if macro == "BEAR":
            return max(1, int(self.router_settings.overrides.bear_trend.breakout_N))
        return max(1, int(self.settings.breakout_lookback_N))

    def _funding_filter_reason(self, row: pd.Series, signal: Signal) -> str | None:
        action = str(row.get("funding_action", "none"))
        if action == "none":
            return None
        if action == "block_long" and signal.side == "LONG":
            return "funding_long"
        if action == "block_short" and signal.side == "SHORT":
            return "funding_short"
        return None

    def _fng_filter_reason(self, row: pd.Series, signal: Signal) -> str | None:
        if not self.fng_filter.enabled:
            return None
        fng_value = row.get("fng_value", np.nan)
        if pd.isna(fng_value):
            return None
        value = float(fng_value)
        if signal.side == "LONG" and value >= float(self.fng_filter.block_long_if_gte):
            return "fng_long"
        if signal.side == "SHORT" and value <= float(self.fng_filter.block_short_if_lte):
            return "fng_short"
        return None

    def _raw_signal_by_mode(self, df: pd.DataFrame, i: int, mode: str) -> tuple[Signal | None, str | None]:
        if mode in ("breakout", "baseline"):
            lookback = self._breakout_lookback_for_row(df.iloc[i])
            if i < lookback + 1:
                return None, "warmup"
            return self._breakout_signal(df, i, lookback)

        if i < 1:
            return None, "warmup"

        if mode == "ema":
            return self._ema_signal(df, i)
        if mode in ("ema_macd", "ml_gate"):
            return self._ema_signal(df, i)
        return None, "unsupported_mode"

    def _mode_filter_reason(self, df: pd.DataFrame, i: int, signal: Signal, mode: str) -> str | None:
        row = df.iloc[i]
        if mode in ("breakout", "baseline"):
            if self.settings.use_rel_volume_filter and row.get("rel_volume_24", 0.0) < self.settings.min_rel_volume:
                return "rel_volume"
            return None

        if mode == "ema":
            return None

        if mode in ("ema_macd", "ml_gate"):
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

            if mode == "ml_gate":
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
        regime = str(row.get("regime", ""))
        micro = regime.split("_")[-1] if "_" in regime else regime
        force_trend_ma = self.router_policy.enabled and micro == "TREND"
        if not self.settings.use_ma200_filter and not force_trend_ma:
            return None

        if signal.side == "SHORT" and not self.router_settings.short_use_ma200_filter:
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

    def _breakout_signal(self, df: pd.DataFrame, i: int, lookback_n: int) -> tuple[Signal | None, str | None]:
        row = df.iloc[i]
        lookback = df.iloc[i - lookback_n : i]
        prev_high = lookback["high"].max()
        prev_low = lookback["low"].min()
        close = row["close"]

        retest_sig = self._maybe_retest_signal(df, i)
        if retest_sig is not None:
            return retest_sig, None

        if close > prev_high:
            if self.settings.retest.enabled:
                self.pending_retest["LONG"] = {
                    "level": float(prev_high),
                    "expires_idx": int(i + self.settings.retest.window_bars),
                }
            return Signal(side="LONG", reason="breakout_high", entry_type="direct"), None
        if close < prev_low:
            if self.settings.retest.enabled:
                self.pending_retest["SHORT"] = {
                    "level": float(prev_low),
                    "expires_idx": int(i + self.settings.retest.window_bars),
                }
            return Signal(side="SHORT", reason="breakout_low", entry_type="direct"), None
        return None, "no_breakout"

    def _maybe_retest_signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        if not self.settings.retest.enabled:
            return None
        row = df.iloc[i]
        atr = float(row.get("atr_14", np.nan))
        if np.isnan(atr):
            return None
        tol = atr * float(self.settings.retest.tolerance_atr)

        for side in ("LONG", "SHORT"):
            pending = self.pending_retest.get(side)
            if not pending:
                continue
            if i > int(pending["expires_idx"]):
                self.pending_retest.pop(side, None)
                continue
            level = float(pending["level"])
            if side == "SHORT":
                touched = float(row["high"]) >= level - tol
                if not touched:
                    continue
                if self.settings.retest.confirmation == "wick_reject":
                    confirmed = float(row["close"]) < float(row["open"]) and float(row["close"]) < level
                else:
                    confirmed = float(row["close"]) < level
                if confirmed:
                    self.pending_retest.pop(side, None)
                    return Signal(side="SHORT", reason="breakout_retest", entry_type="retest")
            else:
                touched = float(row["low"]) <= level + tol
                if not touched:
                    continue
                if self.settings.retest.confirmation == "wick_reject":
                    confirmed = float(row["close"]) > float(row["open"]) and float(row["close"]) > level
                else:
                    confirmed = float(row["close"]) > level
                if confirmed:
                    self.pending_retest.pop(side, None)
                    return Signal(side="LONG", reason="breakout_retest", entry_type="retest")
        return None

    def _mtf_filter_reason(self, row: pd.Series, signal: Signal) -> str | None:
        if not self.mtf_settings.enabled or not self.mtf_settings.require_trend_alignment:
            return None

        regime = str(row.get("regime", ""))
        close_4h = row.get("close_4h", np.nan)
        ema_4h = row.get("ema_200_4h", np.nan)
        slope_4h = row.get("ema_slope_4h", np.nan)
        if pd.isna(close_4h) or pd.isna(ema_4h) or pd.isna(slope_4h):
            return "mtf"

        if signal.side == "SHORT":
            if regime != "BEAR_TREND":
                return "mtf"
            if not (close_4h < ema_4h and slope_4h < 0):
                return "mtf"
            return None

        if signal.side == "LONG":
            if regime != "BULL_TREND":
                return "mtf"
            if not (close_4h > ema_4h and slope_4h > 0):
                return "mtf"
            return None
        return "mtf"

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
