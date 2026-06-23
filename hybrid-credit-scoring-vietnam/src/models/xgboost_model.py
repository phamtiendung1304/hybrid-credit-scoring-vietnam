"""
Hybrid XGBoost Credit Scoring Model
=====================================
The primary model of the Hybrid Credit Scoring project. Combines
traditional CIC bureau features with digital behavioral signals
within an XGBoost gradient boosting framework.

Key Capabilities
-----------------
* Full training pipeline with early stopping on AUC
* Bayesian hyperparameter optimization via Optuna (5-fold stratified CV)
* SHAP-based global and local explainability
* Feature group attribution (bureau vs. behavioral vs. demographic)
* Population Stability Index (PSI) monitoring

Mathematical Foundation
------------------------
XGBoost minimizes the regularized objective:

    ℒ(φ) = Σ_i ℓ(ŷ_i, y_i) + Σ_k Ω(f_k)

where ℓ is binary cross-entropy and:

    Ω(f_k) = γT + ½λ‖w‖²

T = number of leaves, w = leaf weights,
γ = minimum loss reduction (pruning), λ = L2 regularization.

For credit scoring the output probability is:

    PD = σ(raw_score) = 1 / (1 + exp(-raw_score))

References
----------
Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting
    System. KDD '16. https://doi.org/10.1145/2939672.2939785

Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to
    Interpreting Model Predictions. NeurIPS 30.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import optuna
import shap
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from src.models.base_model import BaseCreditModel
from src.features.behavioral_features import compute_behavioral_shap_contribution
from src.utils.config import CONFIG, RANDOM_SEED
from src.utils.logger import logger

optuna.logging.set_verbosity(optuna.logging.WARNING)


class HybridXGBoostModel(BaseCreditModel):
    """XGBoost-based hybrid credit scoring model.

    Parameters
    ----------
    params : dict[str, Any] | None
        XGBoost hyperparameters. If None, values from ``config.yaml``
        are used as defaults.
    random_seed : int
        Global random seed for reproducibility.
    """

    def __init__(
        self,
        params: Optional[dict[str, Any]] = None,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        super().__init__(name="HybridXGBoost", random_seed=random_seed)

        _cfg_params = CONFIG["models"]["xgboost"]
        self.params: dict[str, Any] = params or {
            "n_estimators":      _cfg_params["n_estimators"],
            "max_depth":         _cfg_params["max_depth"],
            "learning_rate":     _cfg_params["learning_rate"],
            "subsample":         _cfg_params["subsample"],
            "colsample_bytree":  _cfg_params["colsample_bytree"],
            "min_child_weight":  _cfg_params["min_child_weight"],
            "reg_alpha":         _cfg_params["reg_alpha"],
            "reg_lambda":        _cfg_params["reg_lambda"],
            "eval_metric":       _cfg_params["eval_metric"],
            "random_state":      random_seed,
            "use_label_encoder": False,
            "verbosity":         0,
        }

        self._shap_values: Optional[np.ndarray] = None
        self._shap_explainer: Optional[shap.TreeExplainer] = None
        self._feature_attribution: Optional[dict[str, float]] = None

    # ─────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "HybridXGBoostModel":
        """Train the XGBoost model with optional early stopping.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Binary default labels for training.
        X_val : pd.DataFrame | None
            Validation feature matrix for early stopping. If None,
            early stopping is disabled.
        y_val : pd.Series | None
            Validation labels. Required when X_val is provided.

        Returns
        -------
        HybridXGBoostModel
            Fitted model.
        """
        self._feature_names = list(X_train.columns)
        n_default = y_train.sum()
        n_non_default = len(y_train) - n_default
        scale_pos_weight = n_non_default / max(n_default, 1)

        fit_params = {**self.params, "scale_pos_weight": scale_pos_weight}
        early_stop = CONFIG["models"]["xgboost"]["early_stopping_rounds"]

        self._model = xgb.XGBClassifier(**fit_params)

        if X_val is not None and y_val is not None:
            self._model.set_params(early_stopping_rounds=early_stop)
            self._model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            best_iter = self._model.best_iteration
            logger.info("Early stopping at iteration {}", best_iter)
        else:
            self._model.fit(X_train, y_train)

        self._is_fitted = True

        train_auc = roc_auc_score(y_train, self._model.predict_proba(X_train)[:, 1])
        logger.success(
            "{} fit complete | n_train={} | train_AUC={:.4f} | scale_pos_weight={:.2f}",
            self.name, len(X_train), train_auc, scale_pos_weight
        )
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted probability of default.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            PD estimates in [0, 1], shape (n_samples,).
        """
        self._check_fitted()
        return self._model.predict_proba(X)[:, 1]

    # ─────────────────────────────────────────────────────────────────────
    # Hyperparameter Optimization
    # ─────────────────────────────────────────────────────────────────────

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 100,
        cv_folds: int = 5,
    ) -> dict[str, Any]:
        """Run Bayesian hyperparameter optimization via Optuna.

        Uses ``StratifiedKFold`` cross-validation to prevent label
        leakage in imbalanced datasets.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Binary training labels.
        n_trials : int
            Number of Optuna trials.
        cv_folds : int
            Number of stratified CV folds.

        Returns
        -------
        dict[str, Any]
            Best hyperparameter configuration found.
        """
        logger.info(
            "Starting Optuna optimization | n_trials={} | cv_folds={}",
            n_trials, cv_folds
        )

        search_space = CONFIG["optuna"]["search_space"]
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_seed)

        def _objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", *search_space["n_estimators"]
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", *search_space["max_depth"]
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *search_space["learning_rate"], log=True
                ),
                "subsample": trial.suggest_float(
                    "subsample", *search_space["subsample"]
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *search_space["colsample_bytree"]
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", *search_space["min_child_weight"]
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", *search_space["reg_alpha"]
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", *search_space["reg_lambda"]
                ),
                "eval_metric": "auc",
                "random_state": self.random_seed,
                "verbosity": 0,
                "use_label_encoder": False,
            }
            clf = xgb.XGBClassifier(**params)
            scores = cross_val_score(
                clf, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_auc = study.best_value

        logger.success(
            "Optuna complete | best_AUC={:.4f} | best_params={}",
            best_auc, best_params
        )

        # Update model params with the best found configuration
        self.params.update(best_params)
        return best_params

    # ─────────────────────────────────────────────────────────────────────
    # Interpretability (SHAP)
    # ─────────────────────────────────────────────────────────────────────

    def compute_shap_values(
        self, X: pd.DataFrame, sample_size: Optional[int] = 500
    ) -> np.ndarray:
        """Compute SHAP values for the provided feature matrix.

        Uses a ``TreeExplainer`` for exact SHAP computation. If
        ``sample_size`` is specified, SHAP is computed on a random
        subsample for speed.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for SHAP computation.
        sample_size : int | None
            If not None, a random subsample of this size is used.

        Returns
        -------
        np.ndarray
            SHAP values of shape ``(n_samples, n_features)``.
        """
        self._check_fitted()

        if sample_size is not None and len(X) > sample_size:
            X = X.sample(n=sample_size, random_state=self.random_seed)

        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self._model)

        shap_values = self._shap_explainer.shap_values(X)
        self._shap_values = shap_values

        # Compute feature group attribution
        self._feature_attribution = compute_behavioral_shap_contribution(
            shap_values=shap_values,
            feature_names=list(X.columns),
        )
        logger.info(
            "SHAP computed | behavioral={:.1f}% | bureau={:.1f}% | demographic={:.1f}%",
            self._feature_attribution["behavioral_pct"],
            self._feature_attribution["bureau_pct"],
            self._feature_attribution["demographic_pct"],
        )
        return shap_values

    def get_feature_attribution(self) -> dict[str, float]:
        """Return SHAP-based feature group attribution.

        Must call ``compute_shap_values()`` first.

        Returns
        -------
        dict[str, float]
            Attribution percentages per feature group.

        Raises
        ------
        RuntimeError
            If SHAP values have not been computed yet.
        """
        if self._feature_attribution is None:
            raise RuntimeError(
                "SHAP values not computed. Call compute_shap_values() first."
            )
        return self._feature_attribution

    def get_top_features(self, n: int = 15) -> pd.DataFrame:
        """Return a ranked table of top features by mean |SHAP| value.

        Parameters
        ----------
        n : int
            Number of top features to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``['feature', 'mean_abs_shap', 'rank']``.
        """
        if self._shap_values is None:
            raise RuntimeError(
                "SHAP values not computed. Call compute_shap_values() first."
            )
        mean_abs = np.abs(self._shap_values).mean(axis=0)
        feature_names = self._feature_names or [f"f{i}" for i in range(len(mean_abs))]

        df = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        df["rank"] = df.index + 1
        return df
