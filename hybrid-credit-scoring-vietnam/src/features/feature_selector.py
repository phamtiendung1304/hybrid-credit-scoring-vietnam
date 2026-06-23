"""
Feature Selector
=================
Implements two complementary feature selection strategies for the
Hybrid Credit Scoring pipeline:

  1. Information Value (IV) Filtering — applied before model training
     to remove statistically non-predictive features.
  2. SHAP-based Importance Pruning — applied post-training to identify
     and optionally remove low-contribution features.

Both strategies are standard in production credit scoring development
(Thomas, 2009; Siddiqi, 2012).

Author : Phạm Tiến Dũng
Project: Hybrid Credit Scoring — NEU Vietnam
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.utils.logger import logger


# ─────────────────────────────────────────────────────────────
# IV Thresholds (industry standard)
# ─────────────────────────────────────────────────────────────

IV_THRESHOLDS: Dict[str, str] = {
    "< 0.02":       "Useless — exclude",
    "0.02 – 0.10":  "Weak — use with caution",
    "0.10 – 0.30":  "Medium — acceptable",
    "0.30 – 0.50":  "Strong — good",
    "> 0.50":        "Suspicious — check for data leakage",
}

IV_MIN_ACCEPTABLE: float = 0.02
IV_LEAKAGE_FLAG:   float = 0.50


# ─────────────────────────────────────────────────────────────
# IV Computation
# ─────────────────────────────────────────────────────────────

def compute_iv(
    series: pd.Series,
    target: pd.Series,
    n_bins: int = 10,
    min_bin_size: int = 50,
) -> Tuple[float, pd.DataFrame]:
    """
    Compute Information Value (IV) and Weight of Evidence (WoE) for a
    single numeric feature relative to a binary target.

    Parameters
    ----------
    series : pd.Series
        Feature values (numeric).
    target : pd.Series
        Binary target (1 = default, 0 = non-default).
    n_bins : int
        Number of equal-frequency bins.
    min_bin_size : int
        Minimum observations per bin; bins below this are merged.

    Returns
    -------
    iv : float
        Total Information Value for the feature.
    woe_table : pd.DataFrame
        Bin-level WoE / IV decomposition table.
    """
    df = pd.DataFrame({"feature": series, "target": target}).dropna()

    if df["feature"].nunique() <= 1:
        return 0.0, pd.DataFrame()

    # Binning (quantile-based for numeric features)
    try:
        df["bin"] = pd.qcut(df["feature"], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["feature"], bins=n_bins)

    grouped = df.groupby("bin")["target"]
    events     = grouped.sum()
    non_events = grouped.count() - events

    total_events     = max(events.sum(), 1)
    total_non_events = max(non_events.sum(), 1)

    dist_events     = events     / total_events
    dist_non_events = non_events / total_non_events

    # Smooth to avoid log(0)
    dist_events     = dist_events.clip(lower=1e-8)
    dist_non_events = dist_non_events.clip(lower=1e-8)

    woe = np.log(dist_events / dist_non_events)
    iv_per_bin = (dist_events - dist_non_events) * woe
    iv_total = iv_per_bin.sum()

    woe_table = pd.DataFrame({
        "bin":               events.index.astype(str),
        "n_events":          events.values,
        "n_non_events":      non_events.values,
        "dist_events":       dist_events.values,
        "dist_non_events":   dist_non_events.values,
        "woe":               woe.values,
        "iv":                iv_per_bin.values,
    })
    return float(iv_total), woe_table


# ─────────────────────────────────────────────────────────────
# IV-based Feature Filter
# ─────────────────────────────────────────────────────────────

class IVFeatureSelector:
    """
    Filters features based on their Information Value (IV) with
    respect to the binary default target.

    Parameters
    ----------
    iv_threshold : float, default 0.02
        Features with IV below this are excluded.
    flag_leakage : bool, default True
        Issue a warning for features with IV > 0.50 (potential leakage).
    n_bins : int, default 10
        Number of bins for IV computation.
    """

    def __init__(
        self,
        iv_threshold: float = IV_MIN_ACCEPTABLE,
        flag_leakage: bool = True,
        n_bins: int = 10,
    ) -> None:
        self.iv_threshold = iv_threshold
        self.flag_leakage = flag_leakage
        self.n_bins = n_bins
        self.iv_summary_: Optional[pd.DataFrame] = None
        self.selected_features_: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "IVFeatureSelector":
        """
        Compute IV for all features and identify those that pass
        the threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (numeric columns only; categoricals should
            be pre-encoded).
        y : pd.Series
            Binary target vector.

        Returns
        -------
        self
        """
        logger.info("IVFeatureSelector.fit() — evaluating {} features", X.shape[1])
        records = []

        for col in X.columns:
            iv, _ = compute_iv(X[col], y, n_bins=self.n_bins)
            status = _iv_label(iv)
            records.append({"feature": col, "iv": iv, "status": status})

            if self.flag_leakage and iv > IV_LEAKAGE_FLAG:
                logger.warning("Feature '{}' has IV={:.4f} > 0.50 — possible data leakage", col, iv)

        self.iv_summary_ = (
            pd.DataFrame(records)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

        self.selected_features_ = (
            self.iv_summary_
            .loc[self.iv_summary_["iv"] >= self.iv_threshold, "feature"]
            .tolist()
        )

        n_removed = X.shape[1] - len(self.selected_features_)
        logger.info(
            "IVFeatureSelector: {} features selected, {} removed (IV < {:.3f})",
            len(self.selected_features_), n_removed, self.iv_threshold,
        )
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return feature matrix with only IV-selected columns."""
        if not self.is_fitted:
            raise RuntimeError("Selector not fitted.")
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def report(self) -> pd.DataFrame:
        """Returns the IV summary table for all evaluated features."""
        if self.iv_summary_ is None:
            raise RuntimeError("Selector not fitted.")
        return self.iv_summary_


# ─────────────────────────────────────────────────────────────
# SHAP-based Post-Training Pruner
# ─────────────────────────────────────────────────────────────

class SHAPFeaturePruner:
    """
    Identifies low-importance features using SHAP mean absolute values
    after the primary model has been trained.

    This is a post-training diagnostic tool, not a pre-training filter.
    Use it to assess model complexity and guide future feature reduction.

    Parameters
    ----------
    importance_threshold : float, default 0.001
        Features with mean |SHAP| below this value are flagged for removal.
    """

    def __init__(self, importance_threshold: float = 0.001) -> None:
        self.importance_threshold = importance_threshold
        self.shap_importance_: Optional[pd.DataFrame] = None
        self.features_to_prune_: Optional[List[str]] = None

    def fit(self, shap_values: np.ndarray, feature_names: List[str]) -> "SHAPFeaturePruner":
        """
        Parameters
        ----------
        shap_values : np.ndarray, shape (n_samples, n_features)
            SHAP values from a fitted explainer.
        feature_names : list of str
            Corresponding feature names.
        """
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        self.shap_importance_ = pd.DataFrame({
            "feature":        feature_names,
            "mean_abs_shap":  mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        self.features_to_prune_ = (
            self.shap_importance_
            .loc[self.shap_importance_["mean_abs_shap"] < self.importance_threshold, "feature"]
            .tolist()
        )

        # Annotate behavioral vs. bureau contribution
        bureau_cols = [f for f in feature_names if f.startswith("cic_")]
        behav_cols  = [f for f in feature_names if not f.startswith("cic_")]

        bureau_shap = self.shap_importance_[self.shap_importance_["feature"].isin(bureau_cols)]["mean_abs_shap"].sum()
        behav_shap  = self.shap_importance_[self.shap_importance_["feature"].isin(behav_cols)]["mean_abs_shap"].sum()
        total_shap  = bureau_shap + behav_shap + 1e-9

        logger.info(
            "SHAP attribution — Bureau: {:.1f}% | Behavioral: {:.1f}%",
            bureau_shap / total_shap * 100,
            behav_shap  / total_shap * 100,
        )
        logger.info("{} features flagged for pruning (mean |SHAP| < {})",
                    len(self.features_to_prune_), self.importance_threshold)
        return self

    def report(self) -> pd.DataFrame:
        """Returns the full SHAP importance table."""
        return self.shap_importance_

    def attribution_by_group(
        self,
        group_map: Dict[str, List[str]],
    ) -> pd.DataFrame:
        """
        Summarizes total SHAP attribution by feature group.

        Parameters
        ----------
        group_map : dict
            Mapping of group name → list of feature names.

        Returns
        -------
        pd.DataFrame with columns: group, total_abs_shap, pct_contribution
        """
        if self.shap_importance_ is None:
            raise RuntimeError("Pruner not fitted.")

        records = []
        total = self.shap_importance_["mean_abs_shap"].sum()
        for group, cols in group_map.items():
            subset = self.shap_importance_[self.shap_importance_["feature"].isin(cols)]
            group_total = subset["mean_abs_shap"].sum()
            records.append({
                "group":            group,
                "total_abs_shap":   group_total,
                "pct_contribution": round(group_total / total * 100, 2),
            })
        return pd.DataFrame(records).sort_values("pct_contribution", ascending=False)


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _iv_label(iv: float) -> str:
    """Maps IV value to interpretability label."""
    if iv < 0.02:
        return "Useless"
    elif iv < 0.10:
        return "Weak"
    elif iv < 0.30:
        return "Medium"
    elif iv < 0.50:
        return "Strong"
    else:
        return "Suspicious (leakage?)"
