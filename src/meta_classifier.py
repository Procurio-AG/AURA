"""
AEW-DSS Phase 4: Meta-Multinomial Stacking Classifier
Fuses terminal/running dynamic Bayesian posteriors with static demographic vectors.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("aew_dss.meta_classifier")


class MetaClassifierWrapper:
    """
    Multinomial Logistic Regression stacking classifier with cost-sensitive
    weighting: omega_k = N / (4 * N_k).
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            C=C,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> MetaClassifierWrapper:
        logger.info("Fitting Meta-Multinomial Logistic Regression layer...")
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
