"""
Model Calibration Diagnostics
================================
Probability calibration is a critical requirement for credit risk models
used in Expected Credit Loss (ECL) calculations under IFRS 9 and for
risk-based pricing. An uncalibrated model may discriminate well (high AUC)
but produce systematically biased probability estimates.

This module provides:
  - Brier Score decomposition (reliability, resolution, uncertainty)
  - Reliability diagram (calibration curve) data
  - Expected Calibration Error (ECE)
  - Platt Scaling and Isotonic Regression calibration wrappers

Reference:
    Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules,
    Prediction, and Estimation. Journal of the American Statistical
    Association, 102(477), 359–378.

Author : Phạm Tiến Dũng
Project: Hybrid Credit Scoring — NEU Vietnam
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Brier Score Decomposition
# ─────────────────────────────────────────────────────────────

def brier_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Decomposes the Brier Score into three components:

        BS = Reliability − Resolution + Uncertainty

    where:
      Reliability (REL) measures calibration error — smaller is better.
      Resolution  (RES) measures sharpness — larger is better.
      Uncertainty (UNC) is a fixed base rate term.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Binary ground-truth labels.
    y_prob : np.ndarray, shape (n_samples,)
        Predicted P(default).
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    dict with keys: brier_score, reliability, resolution, uncertainty
    """
    n = len(y_true)
    base_rate = y_true.mean()
    bs = brier_score_loss(y_true, y_prob)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[1:-1])

    reliability, resolution = 0.0, 0.0
    for k in range(n_bins):
        mask = bin_indices == k
        n_k = mask.sum()
        if n_k == 0:
            continue
        o_k = y_true[mask].mean()   # Observed frequency in bin
        p_k = y_prob[mask].mean()   # Mean predicted probability in bin
        reliability += (n_k / n) * (p_k - o_k) ** 2
        resolution  += (n_k / n) * (o_k - base_rate) ** 2

    uncertainty = base_rate * (1 - base_rate)

    return {
        "brier_score":  float(bs),
        "reliability":  float(reliability),
        "resolution":   float(resolution),
        "uncertainty":  float(uncertainty),
    }


# ─────────────────────────────────────────────────────────────
# Expected Calibration Error
# ─────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE), a scalar measure of
    probability calibration quality.

        ECE = Σ_k (|B_k| / n) × |acc(B_k) − conf(B_k)|

    where B_k is the set of predictions in bin k, acc is the observed
    default rate within the bin, and conf is the mean predicted probability.

    Parameters
    ----------
    y_true : np.ndarray
    y_prob : np.ndarray
    n_bins : int

    Returns
    -------
    float: ECE value in [0, 1] — lower is better.
    """
    n = len(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for k in range(n_bins):
        mask = bin_indices == k
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


# ─────────────────────────────────────────────────────────────
# Reliability Diagram Data
# ─────────────────────────────────────────────────────────────

def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Generates data for a reliability (calibration) diagram.

    Parameters
    ----------
    y_true : np.ndarray
    y_prob : np.ndarray
    n_bins : int

    Returns
    -------
    pd.DataFrame with columns:
        mean_predicted_prob : mean P(default) in bin
        fraction_of_positives: observed default rate in bin
        n_samples: count in bin
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return pd.DataFrame({
        "mean_predicted_prob":   prob_pred,
        "fraction_of_positives": prob_true,
    })


# ─────────────────────────────────────────────────────────────
# Full Calibration Report
# ─────────────────────────────────────────────────────────────

def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Generates a single-row calibration report for a model.

    Parameters
    ----------
    y_true : np.ndarray
    y_prob : np.ndarray
    model_name : str
    n_bins : int

    Returns
    -------
    pd.DataFrame (single row) with calibration metrics.
    """
    decomp = brier_decomposition(y_true, y_prob, n_bins=n_bins)
    ece    = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    mean_prob = float(np.mean(y_prob))
    obs_rate  = float(np.mean(y_true))

    return pd.DataFrame([{
        "model":              model_name,
        "brier_score":        round(decomp["brier_score"], 5),
        "reliability_error":  round(decomp["reliability"], 5),
        "resolution":         round(decomp["resolution"], 5),
        "uncertainty":        round(decomp["uncertainty"], 5),
        "ece":                round(ece, 5),
        "mean_predicted_pd":  round(mean_prob, 5),
        "observed_default_rate": round(obs_rate, 5),
        "calibration_ratio":  round(mean_prob / (obs_rate + 1e-9), 4),
    }])


# ─────────────────────────────────────────────────────────────
# Compare Multiple Models
# ─────────────────────────────────────────────────────────────

def compare_calibration(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Generates a calibration comparison table for multiple models.

    Parameters
    ----------
    results : dict
        Mapping of model_name → (y_true, y_prob).
    n_bins : int

    Returns
    -------
    pd.DataFrame with one row per model.
    """
    rows = []
    for name, (y_true, y_prob) in results.items():
        row = calibration_report(y_true, y_prob, model_name=name, n_bins=n_bins)
        rows.append(row)
    return pd.concat(rows, ignore_index=True)
