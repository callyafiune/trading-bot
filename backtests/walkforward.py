from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtests.baseline import prepare_pattern_frame
from backtests.config import ValidationConfig
from backtests.metrics import write_json
from bot.pattern_mining.feature_engineering import PATTERN_FEATURE_COLUMNS
from bot.pattern_mining.model import train_models


def _iter_walkforward_splits(n_rows: int, train_window: int, test_window: int):
    start = 0
    split_id = 0
    while start + train_window + test_window <= n_rows:
        train_start = start
        train_end = start + train_window
        test_start = train_end
        test_end = test_start + test_window
        yield split_id, train_start, train_end, test_start, test_end
        split_id += 1
        start += test_window


def run_walkforward(df: pd.DataFrame, cfg: ValidationConfig, prepared_df: pd.DataFrame | None = None) -> dict:
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("walkforward requires scikit-learn installed") from exc

    frame = prepared_df.copy() if prepared_df is not None else prepare_pattern_frame(df, cfg)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in PATTERN_FEATURE_COLUMNS if c in frame.columns]
    target = "y_up"
    splits_columns = [
        "split_id",
        "train_start_idx",
        "train_end_idx",
        "test_start_idx",
        "test_end_idx",
        "train_start_ts",
        "train_end_ts",
        "test_start_ts",
        "test_end_ts",
    ]
    metrics_columns = ["split_id", "status", "n_train", "n_test", "auc", "accuracy", "tn", "fp", "fn", "tp", "reason"]

    split_rows = []
    metric_rows = []
    importance_rows = []
    last_conf = None
    skipped_insufficient = 0
    skipped_single_class = 0

    for split_id, tr0, tr1, te0, te1 in _iter_walkforward_splits(
        len(frame),
        cfg.train_window_candles,
        cfg.test_window_candles,
    ):
        train = frame.iloc[tr0:tr1].copy()
        test = frame.iloc[te0:te1].copy()

        train_xy = train[feature_cols + [target]].dropna()
        test_xy = test[feature_cols + [target]].dropna()
        if len(train_xy) < max(100, len(feature_cols) * 5) or len(test_xy) < 20:
            skipped_insufficient += 1
            metric_rows.append(
                {
                    "split_id": split_id,
                    "status": "skipped",
                    "n_train": int(len(train_xy)),
                    "n_test": int(len(test_xy)),
                    "auc": np.nan,
                    "accuracy": np.nan,
                    "tn": np.nan,
                    "fp": np.nan,
                    "fn": np.nan,
                    "tp": np.nan,
                    "reason": "insufficient_rows",
                }
            )
            continue
        if train_xy[target].nunique() < 2 or test_xy[target].nunique() < 2:
            skipped_single_class += 1
            metric_rows.append(
                {
                    "split_id": split_id,
                    "status": "skipped",
                    "n_train": int(len(train_xy)),
                    "n_test": int(len(test_xy)),
                    "auc": np.nan,
                    "accuracy": np.nan,
                    "tn": np.nan,
                    "fp": np.nan,
                    "fn": np.nan,
                    "tp": np.nan,
                    "reason": "single_class_target",
                }
            )
            continue

        n_splits = max(2, min(5, len(train_xy) // 200))
        model_result = train_models(train_xy, feature_cols, target_col=target, n_splits=n_splits)

        proba = model_result.random_forest.predict_proba(test_xy[feature_cols])[:, 1]
        pred = (proba >= 0.5).astype(int)
        y_true = test_xy[target].astype(int).to_numpy()

        auc = float(roc_auc_score(y_true, proba))
        acc = float(accuracy_score(y_true, pred))
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        last_conf = cm

        split_rows.append(
            {
                "split_id": split_id,
                "train_start_idx": tr0,
                "train_end_idx": tr1,
                "test_start_idx": te0,
                "test_end_idx": te1,
                "train_start_ts": str(frame["open_time"].iat[tr0]),
                "train_end_ts": str(frame["open_time"].iat[tr1 - 1]),
                "test_start_ts": str(frame["open_time"].iat[te0]),
                "test_end_ts": str(frame["open_time"].iat[te1 - 1]),
            }
        )
        metric_rows.append(
            {
                "split_id": split_id,
                "n_train": int(len(train_xy)),
                "n_test": int(len(test_xy)),
                "auc": auc,
                "accuracy": acc,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
                "status": "ok",
                "reason": "",
            }
        )

        imp = model_result.feature_importances.copy()
        imp["split_id"] = split_id
        importance_rows.append(imp)

    split_df = pd.DataFrame(split_rows, columns=splits_columns)
    metrics_df = pd.DataFrame(metric_rows, columns=metrics_columns)
    importance_df = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()

    if not importance_df.empty:
        imp_summary = (
            importance_df.groupby("feature")[["decision_tree_importance", "random_forest_importance"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        imp_summary.columns = [
            "feature",
            "decision_tree_importance_mean",
            "decision_tree_importance_std",
            "random_forest_importance_mean",
            "random_forest_importance_std",
        ]
    else:
        imp_summary = pd.DataFrame(
            columns=[
                "feature",
                "decision_tree_importance_mean",
                "decision_tree_importance_std",
                "random_forest_importance_mean",
                "random_forest_importance_std",
            ]
        )

    split_df.to_csv(out_dir / "walkforward_splits.csv", index=False)
    metrics_df.to_csv(out_dir / "walkforward_metrics.csv", index=False)
    imp_summary.to_csv(out_dir / "walkforward_feature_importance.csv", index=False)

    if last_conf is not None:
        conf_df = pd.DataFrame(
            [
                {"actual": 0, "pred_0": int(last_conf[0, 0]), "pred_1": int(last_conf[0, 1])},
                {"actual": 1, "pred_0": int(last_conf[1, 0]), "pred_1": int(last_conf[1, 1])},
            ]
        )
    else:
        conf_df = pd.DataFrame(columns=["actual", "pred_0", "pred_1"])
    conf_df.to_csv(out_dir / "walkforward_last_confusion.csv", index=False)

    metrics_ok = metrics_df[metrics_df["status"] == "ok"] if not metrics_df.empty else pd.DataFrame(columns=metrics_columns)
    mean_auc = float(metrics_ok["auc"].mean()) if not metrics_ok.empty else 0.0
    status_payload = {
        "status": "ok" if not metrics_ok.empty else "no_valid_splits",
        "total_candidate_splits": int(len(list(_iter_walkforward_splits(len(frame), cfg.train_window_candles, cfg.test_window_candles)))),
        "valid_splits": int(len(metrics_ok)),
        "skipped_insufficient_rows": int(skipped_insufficient),
        "skipped_single_class": int(skipped_single_class),
        "mean_auc": mean_auc,
    }
    write_json(out_dir / "walkforward_status.json", status_payload)
    return {
        "splits": split_df,
        "metrics": metrics_df,
        "feature_importance": imp_summary,
        "mean_auc": mean_auc,
        "status": status_payload,
    }
