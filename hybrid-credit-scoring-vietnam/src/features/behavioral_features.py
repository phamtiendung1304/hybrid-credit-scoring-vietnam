"""
Digital Behavioral Features
============================
Constructs and engineers features derived from digital behavioral
signals: mobile banking transaction logs, digital payment records,
and app engagement metrics.

These features represent the key research contribution of the
Hybrid Credit Scoring model — providing an alternative signal source
for Gen Z thin-file customers who lack sufficient CIC bureau history.

Feature Design Rationale
-------------------------
1.  **Transaction regularity over frequency**: The consistency of
    payment behavior (``digital_payment_regularity``) carries more
    predictive signal than raw transaction volume, reflecting that
    financially disciplined borrowers exhibit stable, recurring
    spending patterns irrespective of volume.

2.  **Spending volatility as a risk proxy**: High variance in weekly
    spending amounts (``spending_volatility_90d``) is associated with
    financial instability — a pattern consistent with findings in
    Berg et al. (2020) on digital footprint-based credit scoring.

3.  **Multi-window aggregations**: Transaction counts and amounts are
    aggregated across 30-, 60-, and 90-day windows to capture both
    recency effects (short-window) and behavioral baseline (long-window).

4.  **Merchant diversity as financial maturity indicator**: Spending
    across multiple merchant categories signals a broader, more stable
    financial life compared to highly concentrated spending.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import BEHAVIORAL_FEATURES
from src.utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Core Behavioral Feature Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and engineer behavioral features from the input DataFrame.

    Applies the full behavioral feature engineering pipeline:
      1. Selects base behavioral columns
      2. Constructs composite and ratio features
      3. Applies log transformations to heavy-tailed distributions
      4. Computes temporal momentum features (change between windows)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the raw behavioral signal columns.

    Returns
    -------
    pd.DataFrame
        Engineered behavioral feature matrix, shape (n, m).
    """
    available = [c for c in BEHAVIORAL_FEATURES if c in df.columns]
    X = df[available].copy()

    # ── 1. Log-transform heavy-tailed transaction amount features ─────────
    for col in ["avg_txn_amount_30d", "avg_txn_amount_90d"]:
        if col in X.columns:
            X[f"{col}_log"] = np.log1p(X[col])

    # ── 2. Transaction frequency momentum (30d vs 90d trend) ─────────────
    if "txn_frequency_30d" in X.columns and "txn_frequency_90d" in X.columns:
        # Annualized 30d rate vs 90d baseline: positive → increasing activity
        baseline_30d_from_90d = X["txn_frequency_90d"] / 3.0
        X["txn_momentum"] = (
            (X["txn_frequency_30d"] - baseline_30d_from_90d)
            / (baseline_30d_from_90d.clip(lower=1))
        )

    # ── 3. Average transaction amount momentum ────────────────────────────
    if "avg_txn_amount_30d" in X.columns and "avg_txn_amount_90d" in X.columns:
        X["txn_amount_momentum"] = (
            (X["avg_txn_amount_30d"] - X["avg_txn_amount_90d"])
            / (X["avg_txn_amount_90d"].clip(lower=1))
        )

    # ── 4. Digital payment composite score ───────────────────────────────
    # Combines payment regularity + salary credit + engagement
    if all(c in X.columns for c in [
        "digital_payment_regularity",
        "salary_credit_regularity",
        "mobile_app_engagement_score",
    ]):
        X["digital_financial_health_score"] = (
            0.40 * X["digital_payment_regularity"]
            + 0.35 * X["salary_credit_regularity"]
            + 0.25 * (X["mobile_app_engagement_score"] / 100.0)
        )

    # ── 5. Spending stability index (inverse of volatility) ───────────────
    if "spending_volatility_90d" in X.columns:
        X["spending_stability_index"] = 1.0 / (1.0 + X["spending_volatility_90d"])

    # ── 6. Transaction density (frequency per day equivalent) ─────────────
    if "txn_frequency_90d" in X.columns:
        X["txn_density_daily"] = X["txn_frequency_90d"] / 90.0

    logger.debug("Behavioral features built | shape={}", X.shape)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral Feature Attribution Utility
# ─────────────────────────────────────────────────────────────────────────────

def compute_behavioral_shap_contribution(
    shap_values: np.ndarray,
    feature_names: list[str],
    behavioral_prefixes: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute the proportion of total predictive power from behavioral features.

    Uses mean absolute SHAP values to attribute prediction importance
    to the behavioral vs. traditional feature groups.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP value matrix of shape ``(n_samples, n_features)``.
    feature_names : list[str]
        Column names corresponding to the SHAP value matrix columns.
    behavioral_prefixes : list[str] | None
        List of prefixes identifying behavioral features. Defaults to
        standard prefixes from the project feature schema.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``'behavioral'``, ``'bureau'``, ``'demographic'``,
        and ``'total'``, mapping to their respective mean |SHAP| contributions
        and percentage shares.
    """
    if behavioral_prefixes is None:
        behavioral_prefixes = [
            "txn_", "avg_txn_", "merchant_", "digital_",
            "mobile_app_", "spending_", "salary_credit_",
        ]

    bureau_prefixes = ["cic_"]
    demographic_prefixes = ["age_", "employment_", "declared_", "education_"]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_shap = mean_abs_shap.sum()

    def _group_sum(prefixes: list[str]) -> float:
        return sum(
            mean_abs_shap[i]
            for i, name in enumerate(feature_names)
            if any(name.startswith(p) for p in prefixes)
        )

    behavioral_sum = _group_sum(behavioral_prefixes)
    bureau_sum = _group_sum(bureau_prefixes)
    demographic_sum = _group_sum(demographic_prefixes)

    return {
        "behavioral_mean_abs_shap": round(behavioral_sum, 4),
        "bureau_mean_abs_shap": round(bureau_sum, 4),
        "demographic_mean_abs_shap": round(demographic_sum, 4),
        "total_mean_abs_shap": round(total_shap, 4),
        "behavioral_pct": round(behavioral_sum / total_shap * 100, 1),
        "bureau_pct": round(bureau_sum / total_shap * 100, 1),
        "demographic_pct": round(demographic_sum / total_shap * 100, 1),
    }
