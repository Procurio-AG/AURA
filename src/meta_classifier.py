"""
AURA - Meta-Multinomial Stacking Classifier.
Fuses terminal and running dynamic Bayesian posteriors with static demographic vectors.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression


def stack_features(
    bayesian_posteriors: np.ndarray,
    static_demographics: np.ndarray,
) -> np.ndarray:
    """
    Concatenate dynamic Bayesian posterior distribution with static demographics.

    Args:
        bayesian_posteriors: Array of shape (N, K) representing class probabilities.
        static_demographics: Array of shape (N, D) representing demographic encodings.

    Returns:
        X: Combined feature matrix of shape (N, K + D).
    """
    post = np.asarray(bayesian_posteriors, dtype=np.float64)
    demo = np.asarray(static_demographics, dtype=np.float64)
    if post.shape[0] != demo.shape[0]:
        raise ValueError(
            f"Sample size mismatch: posteriors {post.shape[0]} vs demographics {demo.shape[0]}"
        )
    return np.hstack([post, demo])


class MetaClassifierWrapper:
    """
    Multinomial Logistic Regression stacking classifier with cost-sensitive
    weighting: omega_k = N / (K * N_k).
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            C=C,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MetaClassifierWrapper":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
