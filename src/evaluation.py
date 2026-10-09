"""
AEW-DSS Phase 5: Metric Reporting & Trajectory Visualizations
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logger = logging.getLogger("aew_dss.evaluation")


def evaluate_multiclass_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> Dict:
    if target_names is None:
        target_names = ["Pass", "Withdrawn", "Fail", "Distinction"]

    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    ovr_auc = roc_auc_score(y_true, y_proba, multi_class="ovr")

    logger.info(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    logger.info(f"One-vs-Rest ROC-AUC: {ovr_auc:.4f}")

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "ovr_roc_auc": ovr_auc,
    }


def compute_intervention_priority_scores(proba_matrix: np.ndarray) -> np.ndarray:
    p_withdrawn = proba_matrix[:, 1]
    p_fail = proba_matrix[:, 2]
    p_distinction = proba_matrix[:, 3]
    return 1.0 * p_fail + 0.8 * p_withdrawn - 0.5 * p_distinction
