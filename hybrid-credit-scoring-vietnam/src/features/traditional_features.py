"""
Traditional Bureau Features
============================
Constructs and validates features derived from CIC (Credit Information
Center of Vietnam) bureau data. Includes Weight-of-Evidence (WoE)
transformation and Information Value (IV) computation for feature
selection and Logistic Regression scorecard pipelines.

Mathematical Background
-----------------------
For a binary target y ∈ {0, 1} and a binned categorical variable X:

    WoE_i = ln( P(X=i | y=1) / P(X=i | y=0) )
           = ln( (n_events_i / N_events) / (n_non_events_i / N_non_events) )

    IV = Σ_i (dist_events_i - dist_non_events_i) × WoE_i

IV guidelines (Thomas, 2009):
    IV < 0.02   : not predictive
    0.02–0.10   : weak predictor
    0.10–0.30   : medium predictor
    > 0.30      : strong predictor
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import BUREAU_FEATURES, TARGET_COLUMN
from src.utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# WoE / IV Engine
# ─────────────────────────────────────────────────────────────────────────────

class WoeIvTransformer:
    """Compute Weight-of-Evidence transformation and Information Value.

    This transformer bins continuous features into quantile-based buckets,
    computes WoE and IV for each bucket, and replaces raw feature values
    with their WoE-encoded counterparts.

    Parameters
    ----------
    n_bins : int
        Number of quantile bins for continuous feature binning.
    min_bin_size : float
        Minimum fraction of total observations per bin. Bins smaller
        than this threshold are merged with adjacent bins.
    iv_threshold : float
        Minimum IV for a feature to be retained after transformation.
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_bin_size: float = 0.05,
        iv_threshold: float = 0.02,
    ) -> None:
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.iv_threshold = iv_threshold

        self._bin_edges: dict[str, pd.IntervalIndex] = {}
        self._woe_maps: dict[str, dict[int, float]] = {}
        self._iv_table: dict[str, float] = {}
        self._selected_features: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoeIvTransformer":
        """Fit WoE/IV on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (bureau features only, or any continuous features).
        y : pd.Series
            Binary target vector (0 = non-default, 1 = default).

        Returns
        -------
        WoeIvTransformer
            Fitted transformer (for chaining).
        """
        n_events = y.sum()
        n_non_events = len(y) - n_events

        if n_events == 0 or n_non_events == 0:
            raise ValueError("Target vector must contain both classes (0 and 1).")

        for col in X.columns:
            try:
                iv, woe_map, bin_edges = self._compute_woe_iv(
                    X[col], y, n_events, n_non_events
                )
                self._iv_table[col] = iv
                self._woe_maps[col] = woe_map
                self._bin_edges[col] = bin_edges
            except Exception as exc:
                logger.warning("WoE computation failed for '{}': {}", col, exc)

        self._selected_features = [
            col for col, iv in self._iv_table.items()
            if iv >= self.iv_threshold
        ]

        logger.info(
            "WoE fit complete | total_features={} | selected (IV≥{})={}",
            len(X.columns), self.iv_threshold, len(self._selected_features)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply WoE transformation to X.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to transform.

        Returns
        -------
        pd.DataFrame
            WoE-transformed feature matrix, containing only the
            features that passed the IV threshold during ``fit``.
        """
        result = {}
        for col in self._selected_features:
            if col not in X.columns:
                logger.warning("Column '{}' not found in X during transform.", col)
                continue
            bin_idx = pd.cut(
                X[col], bins=self._bin_edges[col], labels=False, include_lowest=True
            )
            result[f"{col}_woe"] = bin_idx.map(self._woe_maps[col]).fillna(0.0)
        return pd.DataFrame(result, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in a single call."""
        return self.fit(X, y).transform(X)

    def get_iv_table(self) -> pd.DataFrame:
        """Return a sorted IV summary table.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['feature', 'iv', 'strength']``,
            sorted descending by IV.
        """
        rows = []
        for col, iv in sorted(self._iv_table.items(), key=lambda x: -x[1]):
            if iv < 0.02:
                strength = "Not predictive"
            elif iv < 0.10:
                strength = "Weak"
            elif iv < 0.30:
                strength = "Medium"
            else:
                strength = "Strong"
            rows.append({"feature": col, "iv": round(iv, 4), "strength": strength})
        return pd.DataFrame(rows)

    # ── Private helpers ────────────────────────────────────────────────────

    def _compute_woe_iv(
        self,
        series: pd.Series,
        y: pd.Series,
        n_events: int,
        n_non_events: int,
    ) -> tuple[float, dict[int, float], pd.IntervalIndex]:
        """Compute WoE and IV for a single continuous feature."""
        _, bin_edges = pd.qcut(series, q=self.n_bins, retbins=True, duplicates="drop")
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        interval_index = pd.IntervalIndex.from_breaks(bin_edges)

        binned = pd.cut(series, bins=interval_index, labels=False, include_lowest=True)
        df_temp = pd.DataFrame({"bin": binned, "target": y.values})

        agg = df_temp.groupby("bin")["target"].agg(
            n_events=lambda x: x.sum(),
            n_non_events=lambda x: (1 - x).sum(),
        )

        eps = 1e-6
        agg["dist_events"] = (agg["n_events"] + eps) / (n_events + eps)
        agg["dist_non_events"] = (agg["n_non_events"] + eps) / (n_non_events + eps)
        agg["woe"] = np.log(agg["dist_events"] / agg["dist_non_events"])
        agg["iv_contrib"] = (agg["dist_events"] - agg["dist_non_events"]) * agg["woe"]

        iv = agg["iv_contrib"].sum()
        woe_map = agg["woe"].to_dict()

        return iv, woe_map, interval_index


# ─────────────────────────────────────────────────────────────────────────────
# Bureau Feature Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_bureau_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and optionally engineer bureau-derived features.

    Returns the raw bureau feature columns from the input DataFrame,
    with additional derived interaction features appended.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at minimum the BUREAU_FEATURES columns.

    Returns
    -------
    pd.DataFrame
        Bureau feature matrix (raw + engineered), shape (n, m).
    """
    available = [c for c in BUREAU_FEATURES if c in df.columns]
    X = df[available].copy()

    # ── Engineered interaction features ────────────────────────────────
    if "cic_num_delinquencies" in X and "cic_account_age_months" in X:
        # Delinquency rate: normalized by account age
        X["cic_delinquency_rate"] = (
            X["cic_num_delinquencies"]
            / (X["cic_account_age_months"].clip(lower=1))
        )

    if "cic_outstanding_balance_ratio" in X and "cic_credit_utilization" in X:
        # Dual leverage indicator
        X["cic_dual_leverage"] = (
            X["cic_outstanding_balance_ratio"] * X["cic_credit_utilization"]
        )

    if "cic_num_inquiries_6m" in X and "cic_num_active_accounts" in X:
        # Inquiry-to-account ratio: high values signal credit hunger
        X["cic_inquiry_to_account_ratio"] = (
            X["cic_num_inquiries_6m"]
            / (X["cic_num_active_accounts"].clip(lower=1))
        )

    logger.debug("Bureau features built | shape={}", X.shape)
    return X
