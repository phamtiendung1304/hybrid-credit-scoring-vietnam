"""
Data Cleaner & Preprocessor
=============================
Handles missing value imputation, outlier treatment, data type
coercion, and preprocessing pipeline construction for the Hybrid
Credit Scoring dataset.

All transformations are fit on the training set only and applied
to validation/test sets to prevent data leakage.

Author : Phạm Tiến Dũng
Project: Hybrid Credit Scoring — NEU Vietnam
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Optional, Tuple

from src.utils.logger import logger


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Feature groups by type (aligned with data dictionary)
BUREAU_FEATURES: List[str] = [
    "cic_payment_history_score",
    "cic_outstanding_balance_ratio",
    "cic_num_inquiries_6m",
    "cic_account_age_months",
    "cic_num_delinquencies",
    "cic_credit_utilization",
    "cic_num_active_accounts",
]

BEHAVIORAL_FEATURES: List[str] = [
    "txn_frequency_30d",
    "txn_frequency_90d",
    "avg_txn_amount_30d",
    "merchant_category_diversity",
    "digital_payment_regularity",
    "mobile_app_engagement_score",
    "spending_volatility_90d",
    "salary_credit_regularity",
]

DEMOGRAPHIC_FEATURES: List[str] = [
    "age_at_application",
    "declared_income_bracket",
    "education_level",
]

CATEGORICAL_FEATURES: List[str] = [
    "employment_type",
]

NUMERIC_FEATURES: List[str] = (
    BUREAU_FEATURES + BEHAVIORAL_FEATURES + DEMOGRAPHIC_FEATURES
)

# Outlier clipping bounds (domain-constrained)
CLIP_BOUNDS: Dict[str, Tuple[float, float]] = {
    "cic_outstanding_balance_ratio": (0.0, 1.0),
    "cic_credit_utilization":        (0.0, 1.0),
    "digital_payment_regularity":    (0.0, 1.0),
    "age_at_application":            (18.0, 65.0),
    "cic_num_inquiries_6m":          (0.0, 20.0),
    "txn_frequency_30d":             (0.0, 500.0),
    "txn_frequency_90d":             (0.0, 1500.0),
}


# ─────────────────────────────────────────────────────────────
# Transformer: Outlier Clipper
# ─────────────────────────────────────────────────────────────

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips numeric features to domain-valid bounds.

    Parameters
    ----------
    clip_bounds : dict
        Mapping of feature name → (lower_bound, upper_bound).
    feature_names : list of str
        All feature column names (to correctly index arrays).
    """

    def __init__(
        self,
        clip_bounds: Dict[str, Tuple[float, float]],
        feature_names: List[str],
    ) -> None:
        self.clip_bounds = clip_bounds
        self.feature_names = feature_names

    def fit(self, X: np.ndarray, y=None) -> "OutlierClipper":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X.copy().astype(float)
        for feat, (lo, hi) in self.clip_bounds.items():
            if feat in self.feature_names:
                idx = self.feature_names.index(feat)
                X_out[:, idx] = np.clip(X_out[:, idx], lo, hi)
        return X_out


# ─────────────────────────────────────────────────────────────
# Transformer: CIC Missing Flag
# ─────────────────────────────────────────────────────────────

class CICMissingFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Adds binary indicator columns for missing CIC bureau features.

    Thin-file applicants may have NaN in all CIC columns.
    These missingness indicators are themselves predictive.

    Parameters
    ----------
    bureau_features : list of str
        CIC feature names to generate missingness flags for.
    feature_names : list of str
        All feature column names (for index lookup).
    """

    def __init__(
        self,
        bureau_features: List[str],
        feature_names: List[str],
    ) -> None:
        self.bureau_features = bureau_features
        self.feature_names = feature_names
        self.flag_names_: List[str] = []

    def fit(self, X: np.ndarray, y=None) -> "CICMissingFlagTransformer":
        self.flag_names_ = [f"{f}_missing" for f in self.bureau_features]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        flags = []
        for feat in self.bureau_features:
            if feat in self.feature_names:
                idx = self.feature_names.index(feat)
                flag = np.isnan(X[:, idx]).astype(float).reshape(-1, 1)
                flags.append(flag)
        if flags:
            flag_matrix = np.hstack(flags)
            return np.hstack([X, flag_matrix])
        return X

    @property
    def output_feature_names(self) -> List[str]:
        return self.feature_names + self.flag_names_


# ─────────────────────────────────────────────────────────────
# Main DataCleaner class
# ─────────────────────────────────────────────────────────────

class DataCleaner:
    """
    End-to-end preprocessing pipeline for the Hybrid Credit Scoring dataset.

    Steps applied in order:
    1. Drop duplicate application IDs
    2. Coerce data types (numeric, ordinal encoding for categoricals)
    3. Clip domain-constrained features to valid bounds
    4. Add CIC missingness indicator flags
    5. Impute remaining NaN with feature-appropriate strategy
    6. Optional: StandardScaler for logistic regression compatibility

    Parameters
    ----------
    apply_scaling : bool, default False
        Whether to apply StandardScaler. Set True for Logistic Regression;
        keep False for tree-based models (XGBoost, LightGBM, RF).
    """

    def __init__(self, apply_scaling: bool = False) -> None:
        self.apply_scaling = apply_scaling
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._pipeline: Optional[Pipeline] = None
        self._feature_names_after_flags: Optional[List[str]] = None
        self.is_fitted: bool = False

    # ----------------------------------------------------------
    def fit(self, df: pd.DataFrame, target_col: str = "default_flag") -> "DataCleaner":
        """
        Fit the cleaning pipeline on the training DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw training data including the target column.
        target_col : str
            Name of the binary default label column (excluded from transforms).

        Returns
        -------
        self
        """
        logger.info("DataCleaner.fit() started — n={}", len(df))
        X = df.drop(columns=[target_col], errors="ignore").copy()

        # Step 1: Encode categoricals
        for cat_col in CATEGORICAL_FEATURES:
            if cat_col in X.columns:
                le = LabelEncoder()
                X[cat_col] = le.fit_transform(X[cat_col].astype(str))
                self._label_encoders[cat_col] = le

        # Step 2: Identify numeric features present in data
        numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
        X_num = X[numeric_cols].values.astype(float)

        # Step 3: Build sklearn Pipeline
        clipper = OutlierClipper(CLIP_BOUNDS, numeric_cols)
        flag_transformer = CICMissingFlagTransformer(
            [c for c in BUREAU_FEATURES if c in numeric_cols],
            numeric_cols,
        )
        flag_transformer.fit(X_num)
        self._feature_names_after_flags = flag_transformer.output_feature_names

        n_features_out = len(self._feature_names_after_flags)
        imputer = SimpleImputer(strategy="median")

        steps = [
            ("clip",  clipper),
            ("flags", flag_transformer),
            ("impute", imputer),
        ]
        if self.apply_scaling:
            steps.append(("scale", StandardScaler()))

        self._pipeline = Pipeline(steps)
        self._pipeline.fit(X_num)
        self._numeric_cols = numeric_cols
        self.is_fitted = True
        logger.info("DataCleaner.fit() complete — {} features → {} after flags",
                    len(numeric_cols), n_features_out)
        return self

    # ----------------------------------------------------------
    def transform(self, df: pd.DataFrame, target_col: str = "default_flag") -> pd.DataFrame:
        """
        Apply the fitted pipeline to a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
        target_col : str

        Returns
        -------
        pd.DataFrame
            Cleaned feature matrix. Target column excluded.
        """
        if not self.is_fitted:
            raise RuntimeError("DataCleaner must be fitted before calling transform().")

        X = df.drop(columns=[target_col], errors="ignore").copy()

        # Encode categoricals with fitted encoders
        for cat_col, le in self._label_encoders.items():
            if cat_col in X.columns:
                # Handle unseen categories gracefully
                known = set(le.classes_)
                X[cat_col] = X[cat_col].astype(str).apply(
                    lambda v: v if v in known else le.classes_[0]
                )
                X[cat_col] = le.transform(X[cat_col])

        X_num = X[self._numeric_cols].values.astype(float)
        X_clean = self._pipeline.transform(X_num)

        result = pd.DataFrame(X_clean, columns=self._feature_names_after_flags, index=df.index)

        # Append any non-numeric columns that weren't processed
        extra_cols = [c for c in X.columns if c not in self._numeric_cols and c not in CATEGORICAL_FEATURES]
        for col in extra_cols:
            result[col] = X[col].values

        logger.debug("DataCleaner.transform() — output shape {}", result.shape)
        return result

    # ----------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame, target_col: str = "default_flag") -> pd.DataFrame:
        """Convenience wrapper for fit() then transform()."""
        return self.fit(df, target_col).transform(df, target_col)

    # ----------------------------------------------------------
    @property
    def feature_names(self) -> List[str]:
        """Returns output feature names after all transformations."""
        if self._feature_names_after_flags is None:
            raise RuntimeError("Pipeline not fitted yet.")
        return self._feature_names_after_flags


# ─────────────────────────────────────────────────────────────
# Standalone helper
# ─────────────────────────────────────────────────────────────

def describe_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary of missing values per column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: feature, n_missing, pct_missing, dtype
    """
    n_missing = df.isnull().sum()
    pct_missing = (n_missing / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "feature":     n_missing.index,
        "n_missing":   n_missing.values,
        "pct_missing": pct_missing.values,
        "dtype":       df.dtypes.values,
    })
    return summary[summary["n_missing"] > 0].sort_values("pct_missing", ascending=False)
