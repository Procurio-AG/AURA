"""
AURA - Adaptive User Risk Analytics: Dual-Path Sequential Naive Bayes Engine.

Temporal sequential Naive Bayes classifier tracking student performance trajectories
week-by-week. Dynamically alternates between behavior-only likelihoods (Path A) and
joint behavior-assessment likelihoods (Path B) depending on weekly milestone submission status.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.special import logsumexp


class DualPathNaiveBayes:
    """
    Sequential Dual-Path Gaussian Naive Bayes classifier with temporal freezing.

    Tracks weekly posterior probabilities of student outcomes across time steps
    t in [0, T_max]. For weeks without milestone assessment submissions, likelihoods
    are computed using digital behavioral data (Path A). For weeks with milestone
    scores present, joint behavioral and assessment score likelihoods are computed (Path B).
    """

    def __init__(self, num_classes: int = 4, var_smoothing: float = 1e-4) -> None:
        self.num_classes = num_classes
        self.var_smoothing = float(var_smoothing)

        self.classes_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.log_priors_: Optional[np.ndarray] = None
        self.max_weeks_: Optional[int] = None

        self.click_mu_: Optional[np.ndarray] = None
        self.click_sigma2_: Optional[np.ndarray] = None
        self.score_mu_: Optional[np.ndarray] = None
        self.score_sigma2_: Optional[np.ndarray] = None
        self.score_valid_: Optional[np.ndarray] = None

    def fit(
        self,
        weekly_clicks: np.ndarray,
        weekly_scores: np.ndarray,
        labels: np.ndarray,
        priors: Optional[np.ndarray] = None,
    ) -> "DualPathNaiveBayes":
        weekly_clicks = np.asarray(weekly_clicks, dtype=np.float64)
        weekly_scores = np.asarray(weekly_scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        n_samples, n_weeks = weekly_clicks.shape
        if weekly_scores.shape != (n_samples, n_weeks):
            raise ValueError(
                f"Shape mismatch: weekly_clicks is {weekly_clicks.shape}, "
                f"weekly_scores is {weekly_scores.shape}."
            )
        if labels.shape != (n_samples,):
            raise ValueError(
                f"Shape mismatch: labels must be 1D array of length {n_samples}."
            )

        self.max_weeks_ = n_weeks - 1
        self.classes_ = np.arange(self.num_classes, dtype=np.int64)

        if priors is not None:
            priors = np.asarray(priors, dtype=np.float64)
            if priors.shape != (self.num_classes,):
                raise ValueError(
                    f"priors must have shape ({self.num_classes},), got {priors.shape}."
                )
            if not np.isclose(np.sum(priors), 1.0) or np.any(priors < 0.0):
                raise ValueError("priors must be non-negative and sum to 1.0.")
            self.priors_ = priors
        else:
            counts = np.bincount(labels, minlength=self.num_classes).astype(np.float64)
            if np.any(counts == 0):
                raise ValueError(
                    "At least one class has zero samples in the training set."
                )
            self.priors_ = counts / np.sum(counts)

        self.log_priors_ = np.log(self.priors_ + 1e-15)

        self.click_mu_ = np.zeros((self.num_classes, n_weeks), dtype=np.float64)
        self.click_sigma2_ = np.zeros((self.num_classes, n_weeks), dtype=np.float64)
        self.score_mu_ = np.zeros((self.num_classes, n_weeks), dtype=np.float64)
        self.score_sigma2_ = np.zeros((self.num_classes, n_weeks), dtype=np.float64)
        self.score_valid_ = np.zeros((self.num_classes, n_weeks), dtype=bool)

        global_click_mu = np.nanmean(weekly_clicks, axis=0)
        global_click_var = np.nanvar(weekly_clicks, axis=0)

        for k in range(self.num_classes):
            mask_k = labels == k
            n_k = np.sum(mask_k)

            if n_k == 0:
                self.click_mu_[k, :] = global_click_mu
                self.click_sigma2_[k, :] = np.maximum(
                    global_click_var, self.var_smoothing
                )
                continue

            clicks_k = weekly_clicks[mask_k, :]
            scores_k = weekly_scores[mask_k, :]

            mu_c = np.nanmean(clicks_k, axis=0)
            var_c = np.nanvar(clicks_k, axis=0)

            mu_c = np.where(np.isnan(mu_c), global_click_mu, mu_c)
            var_c = np.where(np.isnan(var_c), global_click_var, var_c)
            self.click_mu_[k, :] = mu_c
            self.click_sigma2_[k, :] = np.maximum(var_c, self.var_smoothing)

            for t in range(n_weeks):
                scores_kt = scores_k[:, t]
                valid_scores = scores_kt[~np.isnan(scores_kt)]

                if len(valid_scores) >= 2:
                    s_mu = np.mean(valid_scores)
                    s_var = np.var(valid_scores, ddof=1)
                    self.score_mu_[k, t] = s_mu
                    self.score_sigma2_[k, t] = np.maximum(s_var, self.var_smoothing)
                    self.score_valid_[k, t] = True
                elif len(valid_scores) == 1:
                    self.score_mu_[k, t] = valid_scores[0]
                    self.score_sigma2_[k, t] = self.var_smoothing
                    self.score_valid_[k, t] = True
                else:
                    self.score_mu_[k, t] = 0.0
                    self.score_sigma2_[k, t] = self.var_smoothing
                    self.score_valid_[k, t] = False

        return self

    @staticmethod
    def _gaussian_log_pdf(x: np.ndarray, mu: float, sigma2: float) -> np.ndarray:
        two_pi = 6.283185307179586
        return -0.5 * (
            np.log(two_pi) + np.log(sigma2) + np.square(x - mu) / sigma2
        )

    def _compute_week_log_likelihood(
        self,
        clicks_t: np.ndarray,
        scores_t: np.ndarray,
        t: int,
    ) -> np.ndarray:
        n_samples = clicks_t.shape[0]
        log_L = np.zeros((n_samples, self.num_classes), dtype=np.float64)

        for k in range(self.num_classes):
            mu_c = self.click_mu_[k, t]
            sigma2_c = self.click_sigma2_[k, t]

            log_pdf_clicks = self._gaussian_log_pdf(clicks_t, mu_c, sigma2_c)
            log_pdf_clicks = np.where(np.isnan(clicks_t), 0.0, log_pdf_clicks)

            score_is_valid = self.score_valid_[k, t]
            if score_is_valid:
                mu_s = self.score_mu_[k, t]
                sigma2_s = self.score_sigma2_[k, t]
                log_pdf_scores = self._gaussian_log_pdf(scores_t, mu_s, sigma2_s)
                has_score = ~np.isnan(scores_t)
                log_L[:, k] = log_pdf_clicks + np.where(has_score, log_pdf_scores, 0.0)
            else:
                log_L[:, k] = log_pdf_clicks

        return log_L

    def predict_weekly_posteriors(
        self,
        weekly_clicks: np.ndarray,
        weekly_scores: np.ndarray,
        weights: Optional[np.ndarray] = None,
        unregistration_weeks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.priors_ is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        weekly_clicks = np.asarray(weekly_clicks, dtype=np.float64)
        weekly_scores = np.asarray(weekly_scores, dtype=np.float64)
        n_samples, n_weeks = weekly_clicks.shape

        if weekly_scores.shape != (n_samples, n_weeks):
            raise ValueError("Dimensions of weekly_clicks and weekly_scores must match.")

        if weights is None:
            w = np.ones(n_weeks, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (n_weeks,):
                raise ValueError(
                    f"weights shape {w.shape} does not match week dimension ({n_weeks},)."
                )

        unreg = None
        if unregistration_weeks is not None:
            unreg = np.asarray(unregistration_weeks, dtype=np.float64)
            if unreg.shape != (n_samples,):
                raise ValueError(
                    f"unregistration_weeks must have shape ({n_samples},), got {unreg.shape}."
                )

        posteriors = np.zeros((n_samples, n_weeks, self.num_classes), dtype=np.float64)
        current_log_post = np.tile(self.log_priors_, (n_samples, 1)).astype(np.float64)

        for t in range(n_weeks):
            log_L_t = self._compute_week_log_likelihood(
                weekly_clicks[:, t], weekly_scores[:, t], t
            )

            current_log_post += w[t] * log_L_t

            lse = logsumexp(current_log_post, axis=1, keepdims=True)
            current_log_post -= lse
            p_t = np.exp(current_log_post)

            p_t = p_t / np.sum(p_t, axis=1, keepdims=True)

            if unreg is not None and t > 0:
                freeze_mask = (~np.isnan(unreg)) & (unreg >= 0) & (t > unreg)
                p_t[freeze_mask] = posteriors[freeze_mask, t - 1]
                current_log_post[freeze_mask] = np.log(
                    np.maximum(p_t[freeze_mask], 1e-15)
                )

            posteriors[:, t, :] = p_t

        return posteriors

    def predict_terminal_posteriors(
        self,
        weekly_clicks: np.ndarray,
        weekly_scores: np.ndarray,
        cutoff_week: int,
        weights: Optional[np.ndarray] = None,
        unregistration_weeks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        all_posteriors = self.predict_weekly_posteriors(
            weekly_clicks=weekly_clicks,
            weekly_scores=weekly_scores,
            weights=weights,
            unregistration_weeks=unregistration_weeks,
        )
        if cutoff_week < 0 or cutoff_week >= all_posteriors.shape[1]:
            raise ValueError(
                f"cutoff_week {cutoff_week} out of bounds for max week {all_posteriors.shape[1] - 1}."
            )
        return all_posteriors[:, cutoff_week, :]


def default_milestone_weight_array(
    max_weeks: int = 39,
    assessment_weeks: Optional[List[int]] = None,
    default_weight: float = 1.0,
    milestone_weight: float = 2.5,
) -> np.ndarray:
    if assessment_weeks is None:
        assessment_weeks = [3, 8, 19, 24]

    n_weeks = max_weeks + 1
    weights = np.full(n_weeks, fill_value=default_weight, dtype=np.float64)

    for week in assessment_weeks:
        if 0 <= week < n_weeks:
            weights[week] = milestone_weight

    return weights


def calibrate_milestone_weights(
    val_clicks: np.ndarray,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    grid_search_params: Optional[Dict[str, Union[List[float], List[int], int]]] = None,
    num_classes: int = 4,
    var_smoothing: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, float]]:
    val_clicks = np.asarray(val_clicks, dtype=np.float64)
    val_scores = np.asarray(val_scores, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.int64)

    n_val, n_weeks = val_clicks.shape
    max_weeks = n_weeks - 1

    if grid_search_params is None:
        grid_search_params = {
            "assessment_weeks": [3, 8, 19, 24],
            "base_weights": [0.5, 1.0],
            "milestone_weights": [1.0, 2.0, 3.0, 4.0],
            "eval_cutoff_week": max_weeks,
        }

    assessment_weeks = list(
        grid_search_params.get("assessment_weeks", [3, 8, 19, 24])  # type: ignore
    )
    base_weights = list(grid_search_params.get("base_weights", [1.0]))  # type: ignore
    milestone_weights = list(
        grid_search_params.get("milestone_weights", [1.0, 2.0, 3.0])  # type: ignore
    )
    eval_cutoff = int(grid_search_params.get("eval_cutoff_week", max_weeks))  # type: ignore

    nb = DualPathNaiveBayes(num_classes=num_classes, var_smoothing=var_smoothing)
    nb.fit(val_clicks, val_scores, val_labels)

    best_loss = float("inf")
    best_weights: Optional[np.ndarray] = None
    best_config: Dict[str, float] = {}

    for base_w in base_weights:
        for ms_w in milestone_weights:
            candidate_weights = default_milestone_weight_array(
                max_weeks=max_weeks,
                assessment_weeks=assessment_weeks,
                default_weight=float(base_w),
                milestone_weight=float(ms_w),
            )

            preds = nb.predict_terminal_posteriors(
                val_clicks,
                val_scores,
                cutoff_week=eval_cutoff,
                weights=candidate_weights,
            )

            clipped_preds = np.clip(preds, 1e-15, 1.0 - 1e-15)
            row_indices = np.arange(n_val)
            true_class_probs = clipped_preds[row_indices, val_labels]
            loss = float(-np.mean(np.log(true_class_probs)))

            if loss < best_loss:
                best_loss = loss
                best_weights = candidate_weights
                best_config = {
                    "base_weight": float(base_w),
                    "milestone_weight": float(ms_w),
                    "eval_cutoff_week": float(eval_cutoff),
                    "log_loss": loss,
                }

    if best_weights is None:
        best_weights = np.ones(n_weeks, dtype=np.float64)

    return best_weights, best_config
