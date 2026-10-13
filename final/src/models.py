"""Model ladder: NB baseline, LR primary, LightGBM ceiling, Ridge regressor."""

from __future__ import annotations

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier_pipelines() -> dict[str, Pipeline]:
    return {
        "naive_bayes": Pipeline(
            [("scaler", StandardScaler()), ("clf", GaussianNB())]
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=63,
                        class_weight="balanced",
                        random_state=42,
                        verbosity=-1,
                    ),
                )
            ]
        ),
    }


def build_regressor() -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=42))]
    )


def calibrate(pipeline: Pipeline, X, y) -> CalibratedClassifierCV:
    wrapped = CalibratedClassifierCV(pipeline, method="sigmoid", cv=3)
    wrapped.fit(X, y)
    return wrapped


def classifier_metrics(y_true, p) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "at_risk_recall": float(recall_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def regressor_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def top_drivers(pipeline: Pipeline, row, feature_names, k: int = 5) -> list[dict]:
    """Per-feature contribution to the linear score for a single row."""
    scaler = pipeline.named_steps["scaler"]
    lr = pipeline.named_steps["clf"]
    x = np.asarray(row, dtype="float64").reshape(1, -1)
    z = (x - scaler.mean_) / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    contributions = lr.coef_[0] * z[0]
    order = np.argsort(-np.abs(contributions))[:k]
    return [
        {
            "feature": feature_names[i],
            "contribution": float(contributions[i]),
            "direction": "increases_risk" if contributions[i] > 0 else "decreases_risk",
        }
        for i in order
    ]


def save_bundle(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_bundle(path):
    return joblib.load(path)
