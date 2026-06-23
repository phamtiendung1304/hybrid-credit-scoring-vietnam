"""
Base Credit Model
=================
Abstract base class defining the common interface for all credit
scoring models in the Hybrid Credit Scoring pipeline.

All concrete model implementations (XGBoost, Logistic Regression,
LightGBM, Random Forest) must inherit from ``BaseCreditModel`` and
implement the abstract methods defined here.

Design Pattern
--------------
This follows the Template Method pattern: the base class defines the
algorithm structure (fit → predict_proba → evaluate), while concrete
subclasses implement model-specific training logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.config import RANDOM_SEED, TARGET_COLUMN
from src.utils.logger import logger


class BaseCreditModel(ABC):
    """Abstract base for all credit scoring model wrappers.

    Parameters
    ----------
    name : str
        Human-readable model name (used in logs and reports).
    random_seed : int
        Random seed for reproducibility.
    """

    def __init__(self, name: str, random_seed: int = RANDOM_SEED) -> None:
        self.name = name
        self.random_seed = random_seed
        self._model: Any = None
        self._is_fitted: bool = False
        self._feature_names: Optional[list[str]] = None

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseCreditModel":
        """Train the model on the provided data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training target vector.
        X_val : pd.DataFrame | None
            Optional validation set for early stopping.
        y_val : pd.Series | None
            Optional validation labels.

        Returns
        -------
        BaseCreditModel
            Fitted model (for chaining).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of default (positive class).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples,)`` with PD estimates in [0, 1].
        """
        ...

    # ── Concrete shared methods ────────────────────────────────────────────

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Predict binary default labels at a given probability threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        threshold : float
            Decision boundary. Observations with PD ≥ threshold are
            classified as default (1).

        Returns
        -------
        np.ndarray
            Binary prediction array of shape ``(n_samples,)``.
        """
        self._check_fitted()
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Convenience wrapper returning AUC on the provided data.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series

        Returns
        -------
        float
            AUC-ROC score.
        """
        from sklearn.metrics import roc_auc_score
        self._check_fitted()
        return roc_auc_score(y, self.predict_proba(X))

    def get_feature_names(self) -> list[str]:
        """Return the feature names seen during fitting."""
        self._check_fitted()
        return self._feature_names or []

    # ── Guard ──────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"Model '{self.name}' has not been fitted yet. Call .fit() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', status={status})"
