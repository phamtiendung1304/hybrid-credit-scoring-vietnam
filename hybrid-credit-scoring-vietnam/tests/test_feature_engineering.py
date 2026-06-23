"""
Tests: Feature Engineering
============================
Unit tests for IV computation, WoE transformation, behavioral
feature construction, and feature selection logic.

Run with:
    pytest tests/test_feature_engineering.py -v --cov=src/features

Author : Phạm Tiến Dũng
Project: Hybrid Credit Scoring — NEU Vietnam
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_selector import (
    compute_iv,
    IVFeatureSelector,
    SHAPFeaturePruner,
    _iv_label,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_features() -> tuple:
    """Returns (X, y) with known IV structure."""
    np.random.seed(0)
    n = 2000
    y = pd.Series(np.random.binomial(1, 0.18, n), name="default_flag")

    X = pd.DataFrame({
        # Strong predictor (IV should be > 0.10)
        "strong_feature":   np.where(y == 1, np.random.normal(2, 1, n), np.random.normal(0, 1, n)),
        # Weak predictor (IV ~ 0.02–0.10)
        "weak_feature":     np.where(y == 1, np.random.normal(1, 2, n), np.random.normal(0, 2, n)),
        # Noise (IV < 0.02)
        "noise_feature":    np.random.normal(0, 1, n),
        # Bureau proxy
        "cic_payment_score": np.where(y == 1, np.random.uniform(0, 0.5, n), np.random.uniform(0.4, 1.0, n)),
    })
    return X, y


@pytest.fixture
def shap_values_fixture() -> tuple:
    """Returns synthetic SHAP values and feature names."""
    np.random.seed(1)
    n_samples  = 300
    n_features = 15
    bureau_names   = [f"cic_feat_{i}" for i in range(8)]
    behav_names    = [f"behav_feat_{i}" for i in range(7)]
    feature_names  = bureau_names + behav_names
    shap_values    = np.random.randn(n_samples, n_features)
    # Make bureau features intentionally more important
    shap_values[:, :8] *= 2.0
    return shap_values, feature_names, bureau_names, behav_names


# ─────────────────────────────────────────────────────────────
# compute_iv Tests
# ─────────────────────────────────────────────────────────────

class TestComputeIV:

    def test_returns_float_and_dataframe(self, synthetic_features):
        X, y = synthetic_features
        iv, woe_table = compute_iv(X["strong_feature"], y)
        assert isinstance(iv, float)
        assert isinstance(woe_table, pd.DataFrame)

    def test_strong_feature_high_iv(self, synthetic_features):
        X, y = synthetic_features
        iv, _ = compute_iv(X["strong_feature"], y)
        assert iv > 0.10, f"Strong feature should have IV > 0.10, got {iv:.4f}"

    def test_noise_feature_low_iv(self, synthetic_features):
        X, y = synthetic_features
        iv, _ = compute_iv(X["noise_feature"], y)
        assert iv < 0.10, f"Noise feature should have IV < 0.10, got {iv:.4f}"

    def test_iv_nonnegative(self, synthetic_features):
        X, y = synthetic_features
        for col in X.columns:
            iv, _ = compute_iv(X[col], y)
            assert iv >= 0.0, f"IV must be non-negative, got {iv:.4f} for {col}"

    def test_constant_feature_zero_iv(self, synthetic_features):
        _, y = synthetic_features
        constant = pd.Series(np.ones(len(y)), name="constant")
        iv, _ = compute_iv(constant, y)
        assert iv == 0.0, "Constant feature must have IV = 0"

    def test_woe_table_has_expected_columns(self, synthetic_features):
        X, y = synthetic_features
        _, woe_table = compute_iv(X["strong_feature"], y)
        expected_cols = {"bin", "n_events", "n_non_events", "woe", "iv"}
        assert expected_cols.issubset(set(woe_table.columns))

    def test_iv_bins_sum_to_total(self, synthetic_features):
        X, y = synthetic_features
        iv_total, woe_table = compute_iv(X["strong_feature"], y)
        iv_sum = woe_table["iv"].sum()
        assert abs(iv_total - iv_sum) < 1e-6, "Sum of bin IVs must equal total IV"


# ─────────────────────────────────────────────────────────────
# IVFeatureSelector Tests
# ─────────────────────────────────────────────────────────────

class TestIVFeatureSelector:

    def test_fit_produces_summary(self, synthetic_features):
        X, y = synthetic_features
        selector = IVFeatureSelector(iv_threshold=0.02)
        selector.fit(X, y)
        assert selector.iv_summary_ is not None
        assert len(selector.iv_summary_) == X.shape[1]

    def test_noise_removed(self, synthetic_features):
        X, y = synthetic_features
        selector = IVFeatureSelector(iv_threshold=0.02)
        selector.fit(X, y)
        assert "noise_feature" not in selector.selected_features_, \
            "Noise feature should be removed by IV filter"

    def test_strong_feature_retained(self, synthetic_features):
        X, y = synthetic_features
        selector = IVFeatureSelector(iv_threshold=0.02)
        selector.fit(X, y)
        assert "strong_feature" in selector.selected_features_, \
            "Strong predictor must be retained"

    def test_transform_reduces_columns(self, synthetic_features):
        X, y = synthetic_features
        selector = IVFeatureSelector(iv_threshold=0.05)
        X_selected = selector.fit_transform(X, y)
        assert X_selected.shape[1] <= X.shape[1]

    def test_transform_preserves_rows(self, synthetic_features):
        X, y = synthetic_features
        selector = IVFeatureSelector()
        X_selected = selector.fit_transform(X, y)
        assert len(X_selected) == len(X)

    def test_unfitted_raises(self):
        selector = IVFeatureSelector()
        with pytest.raises(RuntimeError):
            selector.transform(pd.DataFrame({"a": [1, 2]}))

    def test_report_returns_dataframe(self, synthetic_features):
        X, y = synthetic_features
        selector = IVFeatureSelector()
        selector.fit(X, y)
        report = selector.report()
        assert isinstance(report, pd.DataFrame)
        assert "iv" in report.columns


# ─────────────────────────────────────────────────────────────
# SHAPFeaturePruner Tests
# ─────────────────────────────────────────────────────────────

class TestSHAPFeaturePruner:

    def test_fit_produces_importance(self, shap_values_fixture):
        shap_vals, feat_names, bureau, behav = shap_values_fixture
        pruner = SHAPFeaturePruner()
        pruner.fit(shap_vals, feat_names)
        assert pruner.shap_importance_ is not None
        assert len(pruner.shap_importance_) == len(feat_names)

    def test_report_sorted_descending(self, shap_values_fixture):
        shap_vals, feat_names, _, _ = shap_values_fixture
        pruner = SHAPFeaturePruner()
        pruner.fit(shap_vals, feat_names)
        report = pruner.report()
        assert report["mean_abs_shap"].is_monotonic_decreasing, \
            "Report should be sorted by mean_abs_shap descending"

    def test_attribution_by_group_sums_to_100(self, shap_values_fixture):
        shap_vals, feat_names, bureau, behav = shap_values_fixture
        pruner = SHAPFeaturePruner()
        pruner.fit(shap_vals, feat_names)
        attribution = pruner.attribution_by_group({"Bureau": bureau, "Behavioral": behav})
        total_pct = attribution["pct_contribution"].sum()
        assert abs(total_pct - 100.0) < 0.01, \
            f"Attribution percentages should sum to 100, got {total_pct:.2f}"

    def test_bureau_dominates_given_higher_shap(self, shap_values_fixture):
        """Bureau features have 2× higher SHAP values — should have higher attribution."""
        shap_vals, feat_names, bureau, behav = shap_values_fixture
        pruner = SHAPFeaturePruner()
        pruner.fit(shap_vals, feat_names)
        attribution = pruner.attribution_by_group({"Bureau": bureau, "Behavioral": behav})
        bureau_pct = attribution.loc[attribution["group"] == "Bureau", "pct_contribution"].values[0]
        behav_pct  = attribution.loc[attribution["group"] == "Behavioral", "pct_contribution"].values[0]
        assert bureau_pct > behav_pct, "Bureau should dominate given 2× SHAP magnification"


# ─────────────────────────────────────────────────────────────
# IV Label Tests
# ─────────────────────────────────────────────────────────────

class TestIVLabel:

    @pytest.mark.parametrize("iv,expected", [
        (0.00, "Useless"),
        (0.01, "Useless"),
        (0.05, "Weak"),
        (0.15, "Medium"),
        (0.35, "Strong"),
        (0.60, "Suspicious (leakage?)"),
    ])
    def test_iv_labels(self, iv: float, expected: str):
        assert _iv_label(iv) == expected
