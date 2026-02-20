from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.tree import DecisionTreeClassifier
except Exception:  # pragma: no cover
    DecisionTreeClassifier = None
    RandomForestClassifier = None
    TimeSeriesSplit = None


@dataclass
class PatternModelResult:
    decision_tree: DecisionTreeClassifier
    random_forest: RandomForestClassifier
    feature_importances: pd.DataFrame
    cv_scores: dict[str, float]


def _require_sklearn() -> None:
    if DecisionTreeClassifier is None or RandomForestClassifier is None or TimeSeriesSplit is None:
        raise RuntimeError("scikit-learn not available for pattern_mining training")


def train_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> PatternModelResult:
    _require_sklearn()

    train_df = df[feature_cols + [target_col]].copy().dropna()
    if train_df.empty:
        raise ValueError("Not enough data to train pattern_mining")

    X = train_df[feature_cols].astype(float)
    y = train_df[target_col].astype(int)

    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=random_state)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        random_state=random_state,
        n_jobs=-1,
    )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    dt_scores: list[float] = []
    rf_scores: list[float] = []
    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        dt.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        dt_scores.append(float(dt.score(X_test, y_test)))
        rf_scores.append(float(rf.score(X_test, y_test)))

    dt.fit(X, y)
    rf.fit(X, y)

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "decision_tree_importance": dt.feature_importances_,
            "random_forest_importance": rf.feature_importances_,
        }
    ).sort_values("random_forest_importance", ascending=False)

    return PatternModelResult(
        decision_tree=dt,
        random_forest=rf,
        feature_importances=importance_df,
        cv_scores={
            "decision_tree_mean_accuracy": float(np.mean(dt_scores)) if dt_scores else 0.0,
            "random_forest_mean_accuracy": float(np.mean(rf_scores)) if rf_scores else 0.0,
        },
    )
