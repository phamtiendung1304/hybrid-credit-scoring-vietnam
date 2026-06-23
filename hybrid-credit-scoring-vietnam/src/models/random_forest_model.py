"""
Random Forest Credit Scoring Benchmark
=========================================
Bagging ensemble benchmark based on randomized decision trees
(Breiman, 2001). Included as a non-parametric, non-boosting reference
point for assessing the incremental value of gradient boosting.

Reference:
    Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.

Author : Phạm Tiến Dũng
Project: Hybrid Credit Scoring — NEU Vietnam
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, List, Optional

from src.models.base_model import BaseCreditModel
from src.utils.logger import logger


class RandomForestCreditModel(BaseCreditModel):
    """
    Random Forest binary classification model for credit default prediction.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth of each tree. None = unlimited (may overfit).
    min_samples_leaf : int
        Minimum samples required at each leaf node. Increasing this
        reduces overfitting.
    max_features : str or float
        Fraction or method for feature subsampling at each split.
    class_weight : str or dict
        'balanced' automatically adjusts for class imbalance.
    calibrate : bool
        Whether to apply isotonic regression calibration to
        improve probability output reliability.
    random_state : int
    """

    def __init__(
        self,
        n_estimators:    int   = 300,
        max_depth:       Optional[int] = 8,
        min_samples_leaf: int  = 20,
        max_features:    str  = "sqrt",
        class_weight:    str  = "balanced",
        calibrate:       bool = True,
        random_state:    int  = 42,
    ) -> None:
        super().__init__(model_name="RandomForest")
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features     = max_features
        self.class_weight     = class_weight
        self.calibrate        = calibrate
        self.random_state     = random_state

        self._rf: Optional[RandomForestClassifier] = None
        self._calibrated: Optional[CalibratedClassifierCV] = None
        self._feature_names: Optional[List[str]] = None

    # ----------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "RandomForestCreditModel":
        """
        Fit the Random Forest model.

        Parameters
        ----------
        X_train : pd.DataFrame
        y_train : pd.Series
        X_val, y_val : optional — not used (RF has no early stopping)
        """
        logger.info("RandomForestCreditModel.fit() — n_train={}, n_features={}",
                    len(X_train), X_train.shape[1])
        self._feature_names = list(X_train.columns)

        self._rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            n_jobs=-1,
            random_state=self.random_state,
        )

        if self.calibrate:
            self._calibrated = CalibratedClassifierCV(
                self._rf, method="isotonic", cv=5
            )
            self._calibrated.fit(X_train.values, y_train.values)
        else:
            self._rf.fit(X_train.values, y_train.values)

        self.is_fitted = True
        logger.info("RandomForestCreditModel.fit() complete")
        return self

    # ----------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns [P(non-default), P(default)] probabilities."""
        self._check_fitted()
        estimator = self._calibrated if self.calibrate else self._rf
        return estimator.predict_proba(X.values)

    # ----------------------------------------------------------
    def predict_default_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(default) for each observation."""
        return self.predict_proba(X)[:, 1]

    # ----------------------------------------------------------
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns mean decrease in impurity (MDI) feature importance.

        Note: MDI is biased toward high-cardinality features. For a
        fairer assessment, use permutation importance or SHAP values.

        Returns
        -------
        pd.DataFrame with columns: feature, importance
        """
        self._check_fitted()
        if self.calibrate:
            # Average over folds
            importances = np.mean([
                est.estimator.feature_importances_
                for est in self._calibrated.calibrated_classifiers_
            ], axis=0)
        else:
            importances = self._rf.feature_importances_

        return (
            pd.DataFrame({
                "feature":    self._feature_names,
                "importance": importances,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ----------------------------------------------------------
    def get_params(self) -> Dict:
        return {
            "n_estimators":     self.n_estimators,
            "max_depth":        self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features":     self.max_features,
            "class_weight":     self.class_weight,
            "calibrate":        self.calibrate,
            "random_state":     self.random_state,
        }
