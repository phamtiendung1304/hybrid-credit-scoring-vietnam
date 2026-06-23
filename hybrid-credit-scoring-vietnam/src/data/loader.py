"""
Data Loader
===========
Handles ingestion, schema validation, and stratified train/val/test
splitting for the Hybrid Credit Scoring pipeline.

All split operations are fully reproducible: the random seed is read
from ``config.yaml`` and applied to both NumPy and scikit-learn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import CONFIG, RANDOM_SEED, TARGET_COLUMN, resolve_path
from src.utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_COLUMNS: set[str] = {
    # Bureau
    "cic_payment_history_score", "cic_credit_utilization",
    "cic_outstanding_balance_ratio", "cic_num_inquiries_6m",
    "cic_account_age_months", "cic_num_delinquencies", "cic_num_active_accounts",
    # Behavioral
    "txn_frequency_30d", "txn_frequency_60d", "txn_frequency_90d",
    "avg_txn_amount_30d", "avg_txn_amount_90d", "merchant_category_diversity",
    "digital_payment_regularity", "mobile_app_engagement_score",
    "spending_volatility_90d", "salary_credit_regularity",
    # Demographic
    "age_at_application", "employment_type_encoded",
    "declared_income_bracket_encoded", "education_level_encoded",
    # Target
    TARGET_COLUMN,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(file_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load raw data from CSV or Parquet.

    If ``file_path`` is not provided, the function looks for the synthetic
    dataset in ``data/external/synthetic_credit_data.parquet``.

    Parameters
    ----------
    file_path : str | Path | None
        Explicit path to data file (.csv or .parquet). If None, the
        default synthetic dataset is loaded.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all expected columns.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If required columns are missing from the loaded data.
    """
    if file_path is None:
        file_path = resolve_path("data/external/synthetic_credit_data.parquet")
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info("Loading data from {}", file_path)

    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    _validate_schema(df)

    logger.success(
        "Data loaded | shape={} | default_rate={:.2%}",
        df.shape, df[TARGET_COLUMN].mean()
    )
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified train / validation / test split.

    The split is stratified on the target column to preserve the
    default rate across all three partitions.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with target column.
    test_size : float
        Proportion of total data held out for the test set.
    val_size : float
        Proportion of remaining (non-test) data used for validation.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` DataFrames.
    """
    y = df[TARGET_COLUMN]

    df_trainval, df_test = train_test_split(
        df, test_size=test_size, stratify=y, random_state=random_seed
    )
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_size / (1 - test_size),
        stratify=df_trainval[TARGET_COLUMN],
        random_state=random_seed,
    )

    for split, name in [(df_train, "Train"), (df_val, "Val"), (df_test, "Test")]:
        logger.info(
            "{} split | n={} | default_rate={:.2%}",
            name, len(split), split[TARGET_COLUMN].mean()
        )

    return df_train, df_val, df_test


def get_X_y(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing features and the target column.
    feature_cols : list[str] | None
        Columns to include in X. If None, all columns except the
        target are used.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` pair ready for scikit-learn / XGBoost.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_schema(df: pd.DataFrame) -> None:
    """Assert that all expected columns are present in the loaded data."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )
    logger.debug("Schema validation passed | columns={}", list(df.columns))
