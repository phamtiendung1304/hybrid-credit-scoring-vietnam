"""
Logistic Regression Credit Scoring Baseline
=============================================
Implements the regulatory scorecard baseline. Logistic Regression
trained on Weight-of-Evidence (WoE)-transformed features is the
industry standard for credit scoring (Thomas, 2009; Siddiqi, 2012)
and serves as the interpretable benchmark in this project.

The scorecard scaling transforms log-odds output into a conventional
credit score in the [300, 850] range (similar to FICO/VantageScore
conventions), enabling direct business interpretation.

Author : Phạm Tiến Dũng
Project: Hybrid Credit Scoring — NEU Vietnam
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple

from src.models.base_model import BaseCreditModel
from src.utils.logger import logger


# ─────────────────────────────────────────────────────────────
# Scorecard scaling constants (Points-to-Double-Odds convention)
# ─────────────────────────────────────────────────────────────

SCORE_BASE: int   = 600    # Score at base odds
ODDS_BASE:  float = 50.0   # Odds (good:bad) at base score
PDO:        int   = 20     # Points to Double the Odds


class LogisticCreditModel(BaseCreditModel):
    """
    Logistic Regression baseline with WoE feature pipeline.

    Parameters
    ----------
    C : float
        Inverse regularization strength (sklearn convention). Smaller
        values → stronger regularization.
    max_iter : int
        Maximum solver iterations.
    calibrate : bool
        Whether to apply Platt scaling calibration post-training.
    random_state : int
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        calibrate: bool = True,
        random_state: int = 42,
    ) -> None:
        super().__init__(model_name="LogisticRegression")
        self.C            = C
        self.max_iter     = max_iter
        self.calibrate    = calibrate
        self.random_state = random_state

        self._lr: Optional[LogisticRegression] = None
        self._pipeline: Optional[Pipeline] = None
        self._calibrator: Optional[CalibratedClassifierCV] = None
        self._feature_names: Optional[list] = None

    # ----------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LogisticCreditModel":
        """
        Fit the Logistic Regression model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (should be WoE-transformed or cleaned numeric).
        y_train : pd.Series
            Binary target (1 = default).
        X_val, y_val : optional
            Not used; included for interface consistency.
        """
        logger.info("LogisticCreditModel.fit() — n_train={}, n_features={}",
                    len(X_train), X_train.shape[1])
        self._feature_names = list(X_train.columns)

        self._lr = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            class_weight="balanced",
            random_state=self.random_state,
        )

        # Scale features (important for LR convergence)
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     self._lr),
        ])

        if self.calibrate:
            self._calibrator = CalibratedClassifierCV(
                self._pipeline, method="sigmoid", cv=5,
            )
            self._calibrator.fit(X_train.values, y_train.values)
        else:
            self._pipeline.fit(X_train.values, y_train.values)

        self.is_fitted = True
        logger.info("LogisticCreditModel.fit() complete")
        return self

    # ----------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns class probabilities [P(non-default), P(default)].

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
        """
        self._check_fitted()
        estimator = self._calibrator if self.calibrate else self._pipeline
        return estimator.predict_proba(X.values)

    # ----------------------------------------------------------
    def predict_default_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(default) — the credit risk probability score."""
        return self.predict_proba(X)[:, 1]

    # ----------------------------------------------------------
    def predict_credit_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Converts default probability to a credit score on [300, 850].

        Uses the standard scorecard Points-to-Double-Odds (PDO) scaling:

            Factor = PDO / log(2)
            Offset = Score_Base − Factor × log(Odds_Base)
            Score  = Offset − Factor × log(p / (1 − p))

        Higher score → lower credit risk (better borrower).

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of int, shape (n_samples,)
        """
        pd_prob = self.predict_default_proba(X).clip(1e-6, 1 - 1e-6)
        factor = PDO / np.log(2)
        offset = SCORE_BASE - factor * np.log(ODDS_BASE)
        log_odds = np.log(pd_prob / (1 - pd_prob))
        score = offset - factor * log_odds
        return np.clip(score, 300, 850).astype(int)

    # ----------------------------------------------------------
    def get_coefficients(self) -> pd.DataFrame:
        """
        Returns model coefficients for scorecard interpretation.

        Returns
        -------
        pd.DataFrame with columns: feature, coefficient, odds_ratio
        """
        self._check_fitted()

        if self.calibrate:
            # Extract LR from within the calibrated estimator
            lr_coefs = np.mean([
                est.named_steps["lr"].coef_[0]
                for est in self._calibrator.calibrated_classifiers_
            ], axis=0)
        else:
            lr_coefs = self._pipeline.named_steps["lr"].coef_[0]

        return pd.DataFrame({
            "feature":     self._feature_names,
            "coefficient": lr_coefs,
            "odds_ratio":  np.exp(lr_coefs),
        }).sort_values("coefficient", key=abs, ascending=False)

    # ----------------------------------------------------------
    def get_params(self) -> Dict:
        return {
            "C":            self.C,
            "max_iter":     self.max_iter,
            "calibrate":    self.calibrate,
            "random_state": self.random_state,
        }
