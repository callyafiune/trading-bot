from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
except Exception:  # pragma: no cover
    joblib = None
    RandomForestRegressor = None
    HistGradientBoostingRegressor = None
    TimeSeriesSplit = None


@dataclass
class PayoffModelArtifacts:
    model_runup: Any
    model_ddown: Any
    feature_cols: list[str]
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    target_runup_col: str
    target_ddown_col: str


class PayoffPredictor:
    def __init__(self, artifacts: PayoffModelArtifacts) -> None:
        self.artifacts = artifacts

    def predict_payoff(self, current_features_dict: dict) -> tuple[float | None, float | None]:
        feat = pd.DataFrame([current_features_dict])
        for col in self.artifacts.feature_cols:
            if col not in feat.columns:
                feat[col] = np.nan
        X = feat[self.artifacts.feature_cols].copy()
        if X.isna().all(axis=None):
            return None, None
        X = X.fillna(X.mean(numeric_only=True)).fillna(0.0)
        pred_runup = float(max(0.0, self.artifacts.model_runup.predict(X)[0]))
        pred_ddown_abs = float(max(0.0, self.artifacts.model_ddown.predict(X)[0]))
        return pred_runup, pred_ddown_abs


def _require_sklearn() -> None:
    if RandomForestRegressor is None or TimeSeriesSplit is None or joblib is None:
        raise RuntimeError("payoff_model requires scikit-learn and joblib")


def _target_columns(df: pd.DataFrame, horizon: int) -> tuple[str, str]:
    runup_col = f"max_runup_{horizon}h"
    ddown_col = f"max_ddown_{horizon}h"
    if runup_col in df.columns and ddown_col in df.columns:
        return runup_col, ddown_col
    if "max_runup_future" in df.columns and "max_drawdown_future" in df.columns:
        return "max_runup_future", "max_drawdown_future"
    raise ValueError("Required payoff label columns not found")


def train_payoff_models(
    df_features_labels: pd.DataFrame,
    feature_cols: list[str],
    cfg: Any,
    model_dir: str | Path | None = None,
) -> PayoffModelArtifacts:
    _require_sklearn()

    horizon = int(getattr(cfg, "payoff_horizon", getattr(cfg, "horizon_candles", 3)))
    runup_col, ddown_col = _target_columns(df_features_labels, horizon)

    train_df = df_features_labels[feature_cols + [runup_col, ddown_col]].copy().dropna(subset=[runup_col, ddown_col])
    train_df = train_df.fillna(0.0)
    if len(train_df) < max(300, len(feature_cols) * 10):
        raise ValueError("Insufficient rows to train payoff models")

    X = train_df[feature_cols].astype(float)
    y_runup = train_df[runup_col].clip(lower=0.0).astype(float)
    y_ddown_abs = train_df[ddown_col].abs().astype(float)

    rf_runup = RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_leaf=20, random_state=7, n_jobs=-1)
    rf_ddown = RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_leaf=20, random_state=11, n_jobs=-1)

    splitter = TimeSeriesSplit(n_splits=5)
    metric_rows: list[dict[str, Any]] = []

    for split_id, (tr_idx, te_idx) in enumerate(splitter.split(X), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        yru_tr, yru_te = y_runup.iloc[tr_idx], y_runup.iloc[te_idx]
        ydd_tr, ydd_te = y_ddown_abs.iloc[tr_idx], y_ddown_abs.iloc[te_idx]

        ru_model = RandomForestRegressor(n_estimators=80, max_depth=10, min_samples_leaf=20, random_state=100 + split_id, n_jobs=-1)
        dd_model = RandomForestRegressor(n_estimators=80, max_depth=10, min_samples_leaf=20, random_state=200 + split_id, n_jobs=-1)
        ru_model.fit(X_tr, yru_tr)
        dd_model.fit(X_tr, ydd_tr)

        ru_pred = ru_model.predict(X_te)
        dd_pred = dd_model.predict(X_te)

        metric_rows.append(
            {
                "split": split_id,
                "target": "runup",
                "mae": float(mean_absolute_error(yru_te, ru_pred)),
                "rmse": float(np.sqrt(mean_squared_error(yru_te, ru_pred))),
                "r2": float(r2_score(yru_te, ru_pred)),
            }
        )
        metric_rows.append(
            {
                "split": split_id,
                "target": "ddown_abs",
                "mae": float(mean_absolute_error(ydd_te, dd_pred)),
                "rmse": float(np.sqrt(mean_squared_error(ydd_te, dd_pred))),
                "r2": float(r2_score(ydd_te, dd_pred)),
            }
        )

    rf_runup.fit(X, y_runup)
    rf_ddown.fit(X, y_ddown_abs)

    fi = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_runup": rf_runup.feature_importances_,
            "importance_ddown": rf_ddown.feature_importances_,
        }
    )
    fi["importance_mean"] = fi[["importance_runup", "importance_ddown"]].mean(axis=1)
    fi = fi.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    artifacts = PayoffModelArtifacts(
        model_runup=rf_runup,
        model_ddown=rf_ddown,
        feature_cols=feature_cols,
        metrics=pd.DataFrame(metric_rows),
        feature_importance=fi,
        target_runup_col=runup_col,
        target_ddown_col=ddown_col,
    )

    if model_dir is not None:
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf_runup, model_path / "payoff_model_runup.joblib")
        joblib.dump(rf_ddown, model_path / "payoff_model_ddown.joblib")
        joblib.dump(feature_cols, model_path / "payoff_feature_cols.joblib")

    return artifacts


def load_payoff_predictor(models_path: str | Path) -> PayoffPredictor | None:
    _require_sklearn()
    p = Path(models_path)
    runup_p = p / "payoff_model_runup.joblib"
    ddown_p = p / "payoff_model_ddown.joblib"
    cols_p = p / "payoff_feature_cols.joblib"
    if not (runup_p.exists() and ddown_p.exists() and cols_p.exists()):
        return None

    runup_model = joblib.load(runup_p)
    ddown_model = joblib.load(ddown_p)
    feature_cols = list(joblib.load(cols_p))
    artifacts = PayoffModelArtifacts(
        model_runup=runup_model,
        model_ddown=ddown_model,
        feature_cols=feature_cols,
        metrics=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        target_runup_col="",
        target_ddown_col="",
    )
    return PayoffPredictor(artifacts)


def predict_payoff(current_features_dict: dict, predictor: PayoffPredictor | None) -> tuple[float | None, float | None]:
    if predictor is None:
        return None, None
    return predictor.predict_payoff(current_features_dict)
