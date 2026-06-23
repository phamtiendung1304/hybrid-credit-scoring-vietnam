"""
Credit Model Evaluation Metrics
=================================
Comprehensive evaluation framework for binary credit scoring models.

Implements standard banking-grade metrics:
  • AUC-ROC (Area Under Receiver Operating Characteristic Curve)
  • Gini Coefficient  (= 2 × AUC − 1)
  • KS Statistic (Kolmogorov-Smirnov)
  • Brier Score (probabilistic calibration)
  • Log-Loss
  • Population Stability Index (PSI) for deployment monitoring
  • Bootstrap confidence intervals for all metrics

Metric Interpretation (Thomas, 2009)
--------------------------------------
Gini Coefficient:
    < 0.20  : Poor — not suitable for production
    0.20–0.40: Acceptable — minimum bar for deployment
    0.40–0.60: Good — commercial standard
    0.60–0.80: Very good — best-in-class
    > 0.80  : Excellent (possible overfitting — validate carefully)

KS Statistic:
    < 0.20  : Poor
    0.20–0.30: Acceptable
    0.30–0.40: Good
    > 0.40  : Excellent

PSI Threshold:
    < 0.10  : Negligible shift — no action required
    0.10–0.20: Moderate shift — investigate
    > 0.20  : Significant shift — model recalibration required
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
    roc_curve,
)

from src.utils.config import CONFIG, RANDOM_SEED
from src.utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Primary Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_credit_model(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray,
    model_name: str = "Model",
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compute the full suite of credit model evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 = no default, 1 = default).
    y_score : np.ndarray
        Predicted probability of default (PD), in [0, 1].
    model_name : str
        Label for the model in output tables.
    bootstrap_ci : bool
        Whether to compute bootstrap confidence intervals.
    n_bootstrap : int
        Number of bootstrap iterations for CI computation.
    confidence_level : float
        Confidence level for bootstrap intervals (e.g., 0.95 → 95% CI).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all metric columns.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    point_estimates = _compute_point_estimates(y_true, y_score)

    result = {"model": model_name, **point_estimates}

    if bootstrap_ci:
        ci = _bootstrap_confidence_intervals(
            y_true, y_score,
            n_iterations=n_bootstrap,
            confidence_level=confidence_level,
        )
        result.update(ci)

    logger.info(
        "{} | AUC={:.4f} | Gini={:.4f} | KS={:.4f} | Brier={:.4f}",
        model_name,
        point_estimates["auc"],
        point_estimates["gini"],
        point_estimates["ks_statistic"],
        point_estimates["brier_score"],
    )
    return pd.DataFrame([result])


def compare_models(
    results: list[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate individual model evaluation results into a comparison table.

    Parameters
    ----------
    results : list[pd.DataFrame]
        List of DataFrames returned by ``evaluate_credit_model``.

    Returns
    -------
    pd.DataFrame
        Combined comparison table sorted by Gini descending.
    """
    combined = pd.concat(results, ignore_index=True)
    return combined.sort_values("gini", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Individual Metric Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the Gini coefficient (= 2 × AUC − 1).

    Parameters
    ----------
    y_true : np.ndarray
        Binary true labels.
    y_score : np.ndarray
        Predicted default probabilities.

    Returns
    -------
    float
        Gini coefficient in [−1, 1]. Higher is better.
    """
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0


def compute_ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov (KS) statistic.

    KS measures the maximum separation between the cumulative
    distribution of scores for defaulters and non-defaulters.

    Parameters
    ----------
    y_true : np.ndarray
        Binary true labels.
    y_score : np.ndarray
        Predicted default probabilities.

    Returns
    -------
    float
        KS statistic in [0, 1]. Higher is better.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def compute_psi(
    expected_scores: np.ndarray,
    actual_scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI) between two score distributions.

    PSI quantifies how much the score distribution has shifted between
    a reference period (development/training) and a monitoring period
    (deployment). High PSI signals that the model may need recalibration.

    PSI = Σ_i (Actual_i − Expected_i) × ln(Actual_i / Expected_i)

    Parameters
    ----------
    expected_scores : np.ndarray
        Score distribution from the training / reference population.
    actual_scores : np.ndarray
        Score distribution from the deployment / monitoring population.
    n_bins : int
        Number of equal-frequency bins for bucketing.

    Returns
    -------
    float
        PSI value. ≥ 0; higher values indicate greater distributional shift.
    """
    breakpoints = np.quantile(expected_scores, np.linspace(0, 1, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    eps = 1e-6
    expected_pct = np.histogram(expected_scores, bins=breakpoints)[0] / len(expected_scores) + eps
    actual_pct   = np.histogram(actual_scores,   bins=breakpoints)[0] / len(actual_scores)   + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_point_estimates(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Compute all point estimate metrics."""
    return {
        "auc":          round(roc_auc_score(y_true, y_score), 4),
        "gini":         round(compute_gini(y_true, y_score), 4),
        "ks_statistic": round(compute_ks_statistic(y_true, y_score), 4),
        "brier_score":  round(brier_score_loss(y_true, y_score), 4),
        "log_loss":     round(log_loss(y_true, y_score), 4),
    }


def _bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Compute bootstrap CI for AUC and Gini."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, ginis = [], []

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        aucs.append(roc_auc_score(yt, ys))
        ginis.append(2.0 * aucs[-1] - 1.0)

    alpha = 1.0 - confidence_level
    lo, hi = alpha / 2, 1.0 - alpha / 2

    return {
        "auc_ci_lower":  round(float(np.quantile(aucs, lo)), 4),
        "auc_ci_upper":  round(float(np.quantile(aucs, hi)), 4),
        "gini_ci_lower": round(float(np.quantile(ginis, lo)), 4),
        "gini_ci_upper": round(float(np.quantile(ginis, hi)), 4),
    }
