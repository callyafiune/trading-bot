from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.pattern_mining.conditional_stats import build_conditional_stats
from bot.pattern_mining.dataset_builder import build_synchronized_dataset
from bot.pattern_mining.event_detection import detect_events
from bot.pattern_mining.feature_engineering import (
    PATTERN_FEATURE_COLUMNS,
    build_pattern_features,
    build_regime_flags,
    regime_id_from_flags,
)
from bot.pattern_mining.labeling import add_labels
from bot.pattern_mining.model import PatternModelResult, train_models
from bot.pattern_mining.payoff_model import PayoffPredictor, predict_payoff


@dataclass
class EdgeArtifacts:
    horizon: int
    features: list[str]
    model_up: PatternModelResult
    model_down: PatternModelResult
    conditional_stats: pd.DataFrame
    train_matrix: pd.DataFrame
    expected_return_col: str
    mean_map: pd.Series
    std_map: pd.Series


_DEFAULT_SERVICE: "EdgeScoreService | None" = None


class EdgeScoreService:
    def __init__(
        self,
        horizon: int = 3,
        neighbor_count: int = 200,
        *,
        enable_regime_adjustments: bool = True,
        regime_hard_block_id: str = "pa0_sl1_vh1",
        regime_rule_event: str = "break_structure_down",
        regime_prob_boost: float = 0.08,
        regime_cont_prob_mult: float = 0.6,
        regime_expected_return_boost: float = 0.0015,
        payoff_predictor: PayoffPredictor | None = None,
        payoff_fee_bps: float = 4.0,
        payoff_slippage_bps: float = 2.0,
    ) -> None:
        self.horizon = horizon
        self.neighbor_count = max(20, neighbor_count)
        self.artifacts: EdgeArtifacts | None = None
        self.enable_regime_adjustments = bool(enable_regime_adjustments)
        self.regime_hard_block_id = str(regime_hard_block_id)
        self.regime_rule_event = str(regime_rule_event)
        self.regime_prob_boost = float(regime_prob_boost)
        self.regime_cont_prob_mult = float(regime_cont_prob_mult)
        self.regime_expected_return_boost = float(regime_expected_return_boost)
        self.payoff_predictor = payoff_predictor
        self.payoff_fee_bps = float(payoff_fee_bps)
        self.payoff_slippage_bps = float(payoff_slippage_bps)

    def fit(
        self,
        ohlcv: pd.DataFrame,
        oi: pd.DataFrame | None = None,
        cvd: pd.DataFrame | None = None,
        liquidations: pd.DataFrame | None = None,
    ) -> "EdgeScoreService":
        base = build_synchronized_dataset(ohlcv=ohlcv, oi=oi, cvd=cvd, liquidations=liquidations)
        feat = build_pattern_features(base)
        feat = detect_events(feat)
        feat = build_regime_flags(feat)
        feat = add_labels(feat, horizon=self.horizon)

        ret_col = f"future_return_{self.horizon}h"
        train_df = feat.dropna(subset=[ret_col]).copy()
        if train_df.empty:
            self.artifacts = None
            return self

        feature_cols = [c for c in PATTERN_FEATURE_COLUMNS if c in train_df.columns]
        model_up = train_models(train_df, feature_cols, target_col="y_up")
        model_down = train_models(train_df, feature_cols, target_col="y_down")
        cond = build_conditional_stats(train_df, horizon=self.horizon)

        matrix = train_df[feature_cols + [ret_col]].copy().dropna(subset=feature_cols)
        mean_map = matrix[feature_cols].mean()
        std_map = matrix[feature_cols].std(ddof=0).replace(0.0, 1.0)

        self.artifacts = EdgeArtifacts(
            horizon=self.horizon,
            features=feature_cols,
            model_up=model_up,
            model_down=model_down,
            conditional_stats=cond,
            train_matrix=matrix,
            expected_return_col=ret_col,
            mean_map=mean_map,
            std_map=std_map,
        )
        return self

    def _neighbor_expected_return(self, x: pd.DataFrame) -> float:
        assert self.artifacts is not None
        matrix = self.artifacts.train_matrix
        if matrix.empty:
            return 0.0

        features = self.artifacts.features
        x_norm = (x[features].iloc[0] - self.artifacts.mean_map) / self.artifacts.std_map
        m_norm = (matrix[features] - self.artifacts.mean_map) / self.artifacts.std_map
        dists = ((m_norm - x_norm) ** 2).sum(axis=1) ** 0.5

        k = min(self.neighbor_count, len(matrix))
        nearest_idx = dists.nsmallest(k).index
        return float(matrix.loc[nearest_idx, self.artifacts.expected_return_col].mean())

    def get_edge_score(self, current_features: dict) -> dict:
        base = {
            "reversal_prob_3h": 0.5,
            "continuation_prob_3h": 0.5,
            "expected_return_3h": 0.0,
            "regime_id": "pa0_sl0_vh0",
            "regime_price_above_ma99": False,
            "regime_ma99_slope_up": False,
            "regime_vol_high": False,
            "break_structure_down": False,
            "break_structure_up": False,
            "sweep_down_reclaim": False,
            "big_lower_wick": False,
            "pred_runup": None,
            "pred_ddown_abs": None,
            "payoff_expected": None,
            "payoff_ratio": None,
        }
        if self.artifacts is None:
            return base

        row_raw = pd.DataFrame([current_features])
        for col in self.artifacts.features:
            if col not in row_raw.columns:
                row_raw[col] = np.nan
        row = row_raw[self.artifacts.features].copy()
        for col in self.artifacts.features:
            if row[col].isna().all():
                row[col] = float(self.artifacts.mean_map.get(col, 0.0))
            else:
                row[col] = row[col].fillna(float(self.artifacts.mean_map.get(col, 0.0)))

        p_up = float(self.artifacts.model_up.random_forest.predict_proba(row)[0, 1])
        p_down = float(self.artifacts.model_down.random_forest.predict_proba(row)[0, 1])
        exp_ret = self._neighbor_expected_return(row)

        pa = bool(row_raw.get("regime_price_above_ma99", pd.Series([False])).iloc[0])
        sl = bool(row_raw.get("regime_ma99_slope_up", pd.Series([False])).iloc[0])
        vh = bool(row_raw.get("regime_vol_high", pd.Series([False])).iloc[0])
        rid = str(row_raw.get("regime_id", pd.Series([regime_id_from_flags(pa, sl, vh)])).iloc[0])

        event_flags = {
            "break_structure_down": bool(row_raw.get("break_structure_down", pd.Series([False])).iloc[0]),
            "break_structure_up": bool(row_raw.get("break_structure_up", pd.Series([False])).iloc[0]),
            "sweep_down_reclaim": bool(row_raw.get("sweep_down_reclaim", pd.Series([False])).iloc[0]),
            "big_lower_wick": bool(row_raw.get("big_lower_wick", pd.Series([False])).iloc[0]),
        }

        if (
            self.enable_regime_adjustments
            and rid == self.regime_hard_block_id
            and bool(event_flags.get(self.regime_rule_event, False))
        ):
            p_up = max(0.0, min(1.0, p_up * self.regime_cont_prob_mult))
            p_down = max(0.0, min(0.95, p_down + self.regime_prob_boost))
            exp_ret = float(exp_ret + self.regime_expected_return_boost)

        pred_runup, pred_ddown_abs = predict_payoff(current_features, self.payoff_predictor)
        payoff_expected = None
        payoff_ratio = None
        if pred_runup is not None and pred_ddown_abs is not None:
            costs = (self.payoff_fee_bps / 10000.0) + (self.payoff_slippage_bps / 10000.0)
            payoff_expected = float(pred_runup - pred_ddown_abs - costs)
            payoff_ratio = float(pred_runup / max(pred_ddown_abs, 1e-6))

        return {
            "reversal_prob_3h": p_down,
            "continuation_prob_3h": p_up,
            "expected_return_3h": exp_ret,
            "regime_id": rid,
            "regime_price_above_ma99": pa,
            "regime_ma99_slope_up": sl,
            "regime_vol_high": vh,
            "pred_runup": pred_runup,
            "pred_ddown_abs": pred_ddown_abs,
            "payoff_expected": payoff_expected,
            "payoff_ratio": payoff_ratio,
            **event_flags,
        }


def configure_default_edge_service(service: EdgeScoreService | None) -> None:
    global _DEFAULT_SERVICE
    _DEFAULT_SERVICE = service


def get_edge_score(current_features: dict) -> dict:
    if _DEFAULT_SERVICE is None:
        return {
            "reversal_prob_3h": 0.5,
            "continuation_prob_3h": 0.5,
            "expected_return_3h": 0.0,
            "regime_id": "pa0_sl0_vh0",
            "regime_price_above_ma99": False,
            "regime_ma99_slope_up": False,
            "regime_vol_high": False,
            "break_structure_down": False,
            "break_structure_up": False,
            "sweep_down_reclaim": False,
            "big_lower_wick": False,
            "pred_runup": None,
            "pred_ddown_abs": None,
            "payoff_expected": None,
            "payoff_ratio": None,
        }
    return _DEFAULT_SERVICE.get_edge_score(current_features)
