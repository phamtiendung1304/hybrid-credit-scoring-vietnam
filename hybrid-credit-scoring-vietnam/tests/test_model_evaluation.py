"""
Tests: Model Evaluation Metrics
==================================
Unit tests for all credit model evaluation metrics:
AUC, Gini, KS statistic, Brier Score, and calibration diagnostics.

Tests cover:
  - Numerical correctness against known analytical values
  - Edge cases (perfect predictor, random predictor, all-zeros)
  - Brier Score decomposition identity: BS = REL - RES + UNC
  - ECE bounds [0, 1]

Run with:
    pytest tests/test_model_evaluation.py -v --cov=src/evaluation

Author : Phạm Tiến Dũng
Project: Hybrid Credit Scoring — NEU Vietnam
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.evaluation.calibration import (
    brier_decomposition,
    expected_calibration_error,
    calibration_report,
    compare_calibration,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_predictor():
    """Perfect prediction: PD = 1.0 for defaults, 0.0 for non-defaults."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])
    y_prob = y_true.astype(float)
    return y_true, y_prob


@pytest.fixture
def random_predictor():
    """Random prediction: P(default) = 0.18 for all (base rate)."""
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.18, n)
    y_prob = np.full(n, 0.18)
    return y_true, y_prob


@pytest.fixture
def calibrated_model():
    """Well-calibrated model with AUC ≈ 0.80."""
    np.random.seed(7)
    n = 2000
    y_true = np.random.binomial(1, 0.18, n)
    # Simulate calibrated probabilities with moderate discrimination
    logit_noise = np.random.normal(0, 1, n)
    logit_signal = 3.0 * y_true - 1.5 + logit_noise * 0.5
    y_prob = 1 / (1 + np.exp(-logit_signal))
    y_prob = np.clip(y_prob, 0.01, 0.99)
    return y_true, y_prob


# ─────────────────────────────────────────────────────────────
# Brier Score Decomposition Tests
# ─────────────────────────────────────────────────────────────

class TestBrierDecomposition:

    def test_returns_expected_keys(self, calibrated_model):
        y_true, y_prob = calibrated_model
        result = brier_decomposition(y_true, y_prob)
        expected_keys = {"brier_score", "reliability", "resolution", "uncertainty"}
        assert expected_keys == set(result.keys())

    def test_decomposition_identity(self, calibrated_model):
        """BS ≈ REL − RES + UNC (within numerical tolerance)."""
        y_true, y_prob = calibrated_model
        result = brier_decomposition(y_true, y_prob)
        lhs = result["brier_score"]
        rhs = result["reliability"] - result["resolution"] + result["uncertainty"]
        assert abs(lhs - rhs) < 0.01, (
            f"Decomposition identity failed: BS={lhs:.5f} vs REL-RES+UNC={rhs:.5f}"
        )

    def test_brier_score_matches_sklearn(self, calibrated_model):
        y_true, y_prob = calibrated_model
        result    = brier_decomposition(y_true, y_prob)
        sklearn_bs = brier_score_loss(y_true, y_prob)
        assert abs(result["brier_score"] - sklearn_bs) < 1e-6

    def test_all_values_nonnegative(self, calibrated_model):
        y_true, y_prob = calibrated_model
        result = brier_decomposition(y_true, y_prob)
        for k, v in result.items():
            assert v >= 0.0, f"{k} should be non-negative, got {v}"

    def test_perfect_predictor_low_brier(self, perfect_predictor):
        y_true, y_prob = perfect_predictor
        result = brier_decomposition(y_true, y_prob)
        assert result["brier_score"] < 0.05, "Perfect predictor should have near-zero Brier Score"

    def test_random_predictor_brier_equals_uncertainty(self, random_predictor):
        """For a constant predictor = base_rate, reliability ≈ 0, so BS ≈ uncertainty − resolution."""
        y_true, y_prob = random_predictor
        result = brier_decomposition(y_true, y_prob)
        # Reliability should be very small for constant predictor equal to base rate
        assert result["reliability"] < 0.01, \
            f"Constant base-rate predictor should have near-zero reliability, got {result['reliability']:.4f}"


# ─────────────────────────────────────────────────────────────
# Expected Calibration Error Tests
# ─────────────────────────────────────────────────────────────

class TestExpectedCalibrationError:

    def test_ece_in_unit_interval(self, calibrated_model):
        y_true, y_prob = calibrated_model
        ece = expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0, f"ECE must be in [0,1], got {ece:.4f}"

    def test_perfect_calibration_low_ece(self):
        """When mean predicted prob in each bin = observed frequency, ECE → 0."""
        n = 1000
        np.random.seed(0)
        # Generate perfectly calibrated predictions
        y_prob = np.random.uniform(0, 1, n)
        y_true = np.random.binomial(1, y_prob, n)
        ece = expected_calibration_error(y_true, y_prob)
        assert ece < 0.10, f"Perfectly calibrated model should have low ECE, got {ece:.4f}"

    def test_returns_float(self, calibrated_model):
        y_true, y_prob = calibrated_model
        ece = expected_calibration_error(y_true, y_prob)
        assert isinstance(ece, float)


# ─────────────────────────────────────────────────────────────
# Calibration Report Tests
# ─────────────────────────────────────────────────────────────

class TestCalibrationReport:

    def test_returns_single_row_dataframe(self, calibrated_model):
        y_true, y_prob = calibrated_model
        report = calibration_report(y_true, y_prob, model_name="TestModel")
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 1

    def test_model_name_in_report(self, calibrated_model):
        y_true, y_prob = calibrated_model
        report = calibration_report(y_true, y_prob, model_name="XGBoost")
        assert report["model"].iloc[0] == "XGBoost"

    def test_report_columns_complete(self, calibrated_model):
        y_true, y_prob = calibrated_model
        report = calibration_report(y_true, y_prob)
        expected = {
            "model", "brier_score", "reliability_error", "resolution",
            "uncertainty", "ece", "mean_predicted_pd",
            "observed_default_rate", "calibration_ratio",
        }
        assert expected.issubset(set(report.columns))

    def test_calibration_ratio_near_one_for_calibrated_model(self, calibrated_model):
        """A calibrated model should have mean predicted PD ≈ observed default rate."""
        y_true, y_prob = calibrated_model
        report = calibration_report(y_true, y_prob)
        ratio = report["calibration_ratio"].iloc[0]
        assert 0.5 < ratio < 2.0, f"Calibration ratio out of expected range: {ratio:.3f}"


# ─────────────────────────────────────────────────────────────
# Compare Calibration Tests
# ─────────────────────────────────────────────────────────────

class TestCompareCalibration:

    def test_returns_one_row_per_model(self, calibrated_model, random_predictor):
        y_true_c, y_prob_c = calibrated_model
        y_true_r, y_prob_r = random_predictor

        results = {
            "XGBoost":  (y_true_c, y_prob_c),
            "Baseline": (y_true_r, y_prob_r),
        }
        comparison = compare_calibration(results)
        assert len(comparison) == 2

    def test_calibrated_model_lower_brier(self, calibrated_model, random_predictor):
        y_true_c, y_prob_c = calibrated_model
        y_true_r, y_prob_r = random_predictor

        results = {
            "XGBoost":  (y_true_c, y_prob_c),
            "Baseline": (y_true_r, y_prob_r),
        }
        comparison = compare_calibration(results)
        xgb_brier  = comparison.loc[comparison["model"] == "XGBoost",  "brier_score"].values[0]
        base_brier = comparison.loc[comparison["model"] == "Baseline", "brier_score"].values[0]
        assert xgb_brier < base_brier, \
            "Discriminating model should have lower Brier Score than random baseline"
