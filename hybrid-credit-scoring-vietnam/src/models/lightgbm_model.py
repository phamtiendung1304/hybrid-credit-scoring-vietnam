"""
LightGBM Credit Scoring Benchmark
===================================
Histogram-based gradient boosted decision tree benchmark.
LightGBM is included as a secondary GBDT comparator to XGBoost,
offering faster training via the leaf-wise growth strategy (Ke et al., 2017).

Results from this model are included in the model comparison table
in the research report.

Reference:
    Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient
    Boosting Decision Tree. NeurIPS 2017.

Author : Phạm Tiến Dũng
Project: Hybrid Credit Scoring — NEU Vietnam
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Optional, Tuple

from src.models.base_model import BaseCreditModel
from src.utils.logger import logger


class LightGBMCreditModel(BaseCreditModel):
    """
    LightGBM binary classification model for credit default prediction.

    Parameters
    ----------
    n_estimators : int
    max_depth : int
    learning_rate : float
    subsample : float
    colsample_bytree : float
    reg_alpha : float       L1 regularization
    reg_lambda : float      L2 regularization
    scale_pos_weight : float  Weight of positive (default) class
    random_state : int
    early_stopping_rounds : int
        Stops training if validation AUC does not improve.
    """

    def __init__(
        self,
        n_estimators:         int   = 500,
        max_depth:            int   = 6,
        learning_rate:        float = 0.05,
        subsample:            float = 0.8,
        colsample_bytree:     float = 0.8,
        reg_alpha:            float = 0.1,
        reg_lambda:           float = 1.0,
        scale_pos_weight:     float = 5.0,
        random_state:         int   = 42,
        early_stopping_rounds: int  = 50,
    ) -> None:
        super().__init__(model_name="LightGBM")
        self.n_estimators          = n_estimators
        self.max_depth             = max_depth
        self.learning_rate         = learning_rate
        self.subsample             = subsample
        self.colsample_bytree      = colsample_bytree
        self.reg_alpha             = reg_alpha
        self.reg_lambda            = reg_lambda
        self.scale_pos_weight      = scale_pos_weight
        self.random_state          = random_state
        self.early_stopping_rounds = early_stopping_rounds

        self._model: Optional[lgb.LGBMClassifier] = None
        self._feature_names: Optional[List[str]] = None
        self._best_iteration: Optional[int] = None

    # ----------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMCreditModel":
        """
        Fit LightGBM. If validation data is provided, early stopping is applied.

        Parameters
        ----------
        X_train : pd.DataFrame
        y_train : pd.Series
        X_val : pd.DataFrame, optional
        y_val : pd.Series, optional
        """
        logger.info("LightGBMCreditModel.fit() — n_train={}, n_features={}",
                    len(X_train), X_train.shape[1])
        self._feature_names = list(X_train.columns)

        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            objective="binary",
            metric="auc",
            verbose=-1,
        )

        fit_kwargs: Dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val.values, y_val.values)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),
            ]

        self._model.fit(X_train.values, y_train.values, **fit_kwargs)

        if hasattr(self._model, "best_iteration_"):
            self._best_iteration = self._model.best_iteration_
            logger.info("LightGBM best iteration: {}", self._best_iteration)

        self.is_fitted = True
        logger.info("LightGBMCreditModel.fit() complete")
        return self

    # ----------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns [P(non-default), P(default)] probabilities."""
        self._check_fitted()
        return self._model.predict_proba(X.values)

    # ----------------------------------------------------------
    def predict_default_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(default) for each observation."""
        return self.predict_proba(X)[:, 1]

    # ----------------------------------------------------------
    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Returns LightGBM native feature importance.

        Parameters
        ----------
        importance_type : str
            'gain' (default) — average gain per split; more reliable than 'split'.
            'split' — number of times feature appears in a split.

        Returns
        -------
        pd.DataFrame with columns: feature, importance
        """
        self._check_fitted()
        importance = self._model.feature_importances_
        return (
            pd.DataFrame({
                "feature":    self._feature_names,
                "importance": importance,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ----------------------------------------------------------
    def get_params(self) -> Dict:
        return {
            "n_estimators":          self.n_estimators,
            "max_depth":             self.max_depth,
            "learning_rate":         self.learning_rate,
            "subsample":             self.subsample,
            "colsample_bytree":      self.colsample_bytree,
            "reg_alpha":             self.reg_alpha,
            "reg_lambda":            self.reg_lambda,
            "scale_pos_weight":      self.scale_pos_weight,
            "random_state":          self.random_state,
        }
