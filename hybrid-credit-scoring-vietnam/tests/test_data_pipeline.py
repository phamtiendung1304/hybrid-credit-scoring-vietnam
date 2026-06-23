"""
Tests: Data Pipeline
======================
Unit tests for data loading, cleaning, validation, and synthetic
data generation. Uses pytest with isolated fixtures to ensure
all transforms are stateless and reproducible.

Run with:
    pytest tests/test_data_pipeline.py -v --cov=src/data

Author : Phạm Tiến Dũng
Project: Hybrid Credit Scoring — NEU Vietnam
"""

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import (
    DataCleaner,
    BUREAU_FEATURES,
    BEHAVIORAL_FEATURES,
    DEMOGRAPHIC_FEATURES,
    describe_missing,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Generates a minimal synthetic DataFrame matching the project schema."""
    np.random.seed(42)
    n = 500
    data = {
        # Bureau features
        "cic_payment_history_score":    np.random.uniform(0, 1, n),
        "cic_outstanding_balance_ratio": np.random.uniform(0, 1.2, n),  # intentionally > 1 to test clipping
        "cic_num_inquiries_6m":          np.random.randint(0, 25, n).astype(float),
        "cic_account_age_months":        np.random.randint(0, 120, n).astype(float),
        "cic_num_delinquencies":         np.random.randint(0, 10, n).astype(float),
        "cic_credit_utilization":        np.random.uniform(-0.1, 1.1, n),  # intentionally out-of-range
        "cic_num_active_accounts":       np.random.randint(0, 10, n).astype(float),
        # Behavioral features
        "txn_frequency_30d":             np.random.randint(0, 600, n).astype(float),
        "txn_frequency_90d":             np.random.randint(0, 2000, n).astype(float),
        "avg_txn_amount_30d":            np.random.uniform(10_000, 5_000_000, n),
        "merchant_category_diversity":   np.random.randint(1, 30, n).astype(float),
        "digital_payment_regularity":    np.random.uniform(-0.1, 1.2, n),
        "mobile_app_engagement_score":   np.random.uniform(0, 10, n),
        "spending_volatility_90d":       np.random.uniform(0, 5_000_000, n),
        "salary_credit_regularity":      np.random.randint(0, 2, n).astype(float),
        # Demographic
        "age_at_application":            np.random.randint(16, 70, n).astype(float),  # outside [18,65] intentional
        "declared_income_bracket":       np.random.randint(1, 5, n).astype(float),
        "education_level":               np.random.randint(1, 5, n).astype(float),
        # Categorical
        "employment_type":               np.random.choice(["formal", "informal", "self_employed", "student"], n),
        # Target
        "default_flag":                  np.random.randint(0, 2, n),
    }
    df = pd.DataFrame(data)

    # Introduce 15% missingness in CIC features (thin-file simulation)
    for col in BUREAU_FEATURES:
        mask = np.random.rand(n) < 0.15
        df.loc[mask, col] = np.nan

    return df


@pytest.fixture
def cleaner(sample_df) -> DataCleaner:
    """Returns a fitted DataCleaner instance."""
    dc = DataCleaner(apply_scaling=False)
    dc.fit(sample_df, target_col="default_flag")
    return dc


# ─────────────────────────────────────────────────────────────
# DataCleaner Tests
# ─────────────────────────────────────────────────────────────

class TestDataCleaner:

    def test_fit_sets_fitted_flag(self, cleaner: DataCleaner):
        assert cleaner.is_fitted is True

    def test_transform_returns_dataframe(self, cleaner: DataCleaner, sample_df: pd.DataFrame):
        result = cleaner.transform(sample_df, target_col="default_flag")
        assert isinstance(result, pd.DataFrame), "Output must be a DataFrame"

    def test_no_missing_values_after_transform(self, cleaner: DataCleaner, sample_df: pd.DataFrame):
        result = cleaner.transform(sample_df, target_col="default_flag")
        n_missing = result.isnull().sum().sum()
        assert n_missing == 0, f"Transform should produce no NaN; found {n_missing}"

    def test_target_column_excluded(self, cleaner: DataCleaner, sample_df: pd.DataFrame):
        result = cleaner.transform(sample_df, target_col="default_flag")
        assert "default_flag" not in result.columns, "Target column must not appear in output"

    def test_clip_bounds_applied(self, cleaner: DataCleaner, sample_df: pd.DataFrame):
        """Check that clipped features are within [0, 1] after transform."""
        result = cleaner.transform(sample_df, target_col="default_flag")
        for col in ["cic_outstanding_balance_ratio", "cic_credit_utilization"]:
            if col in result.columns:
                assert result[col].min() >= 0.0, f"{col} below 0 after clipping"
                assert result[col].max() <= 1.0, f"{col} above 1 after clipping"

    def test_missingness_flags_created(self, cleaner: DataCleaner, sample_df: pd.DataFrame):
        """CIC missingness indicator columns should be present."""
        result = cleaner.transform(sample_df, target_col="default_flag")
        flag_cols = [c for c in result.columns if c.endswith("_missing")]
        assert len(flag_cols) > 0, "No missingness flag columns found"

    def test_feature_names_property(self, cleaner: DataCleaner):
        names = cleaner.feature_names
        assert isinstance(names, list)
        assert len(names) > 0

    def test_unfitted_transform_raises(self):
        dc = DataCleaner()
        with pytest.raises(RuntimeError, match="must be fitted"):
            dc.transform(pd.DataFrame({"a": [1, 2]}))

    def test_fit_transform_equivalence(self, sample_df: pd.DataFrame):
        """fit_transform() must produce the same result as fit() + transform()."""
        dc1 = DataCleaner()
        dc2 = DataCleaner()
        result1 = dc1.fit(sample_df, "default_flag").transform(sample_df, "default_flag")
        result2 = dc2.fit_transform(sample_df, "default_flag")
        pd.testing.assert_frame_equal(result1, result2)

    def test_no_leakage_train_test(self, sample_df: pd.DataFrame):
        """Fitting on train and transforming test must not raise or leak info."""
        train = sample_df.iloc[:400].copy()
        test  = sample_df.iloc[400:].copy()
        dc = DataCleaner()
        dc.fit(train, "default_flag")
        result = dc.transform(test, "default_flag")
        assert len(result) == len(test)
        assert result.isnull().sum().sum() == 0


# ─────────────────────────────────────────────────────────────
# describe_missing Tests
# ─────────────────────────────────────────────────────────────

class TestDescribeMissing:

    def test_returns_dataframe(self, sample_df: pd.DataFrame):
        result = describe_missing(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns_correct(self, sample_df: pd.DataFrame):
        result = describe_missing(sample_df)
        expected = {"feature", "n_missing", "pct_missing", "dtype"}
        assert expected.issubset(set(result.columns))

    def test_only_missing_features_returned(self, sample_df: pd.DataFrame):
        result = describe_missing(sample_df)
        assert (result["n_missing"] > 0).all(), "Only features with missing values should be returned"

    def test_empty_dataframe_no_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = describe_missing(df)
        assert len(result) == 0, "No missing values — result should be empty"
