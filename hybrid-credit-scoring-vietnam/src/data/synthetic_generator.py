"""
Synthetic Data Generator
========================
Generates a synthetic dataset that replicates the statistical
properties of the proprietary Vietnamese banking dataset used in
the Hybrid Credit Scoring project.

The generator preserves:
  - Realistic marginal distributions for all feature groups
  - Cross-feature correlations (bureau ↔ behavioral)
  - The ~17.5% default rate observed in the original sample
  - The thin-file profile of Gen Z applicants

Usage
-----
    python -m src.data.synthetic_generator --n_samples 15000 --seed 42

    or programmatically:

    from src.data.synthetic_generator import generate_synthetic_dataset
    df = generate_synthetic_dataset(n_samples=15000, random_seed=42)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.utils.config import CONFIG, resolve_path
from src.utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    n_samples: int = 15_000,
    random_seed: int = 42,
    default_rate: float = 0.175,
) -> pd.DataFrame:
    """Generate a synthetic credit application dataset.

    Produces bureau features, digital behavioral features, demographic
    features, and a binary default label for ``n_samples`` observations.
    The default flag is derived from a latent risk score so that the
    joint feature-label distribution is statistically coherent.

    Parameters
    ----------
    n_samples : int
        Number of synthetic observations to generate.
    random_seed : int
        NumPy random seed for full reproducibility.
    default_rate : float
        Target proportion of positive-class (default) observations.

    Returns
    -------
    pd.DataFrame
        DataFrame with shape ``(n_samples, n_features + 1)`` where the
        last column is ``default_flag`` (0 = no default, 1 = default).
    """
    rng = np.random.default_rng(random_seed)
    logger.info("Generating synthetic dataset | n={} | seed={}", n_samples, random_seed)

    # ── Latent risk score (drives correlations with target) ───────────────
    latent_risk = rng.standard_normal(n_samples)  # higher → higher default risk

    # ── Bureau Features ───────────────────────────────────────────────────
    # Payment history: high score = good (inversely correlated with risk)
    cic_payment_history_score = np.clip(
        700 - 80 * latent_risk + rng.normal(0, 30, n_samples), 300, 850
    ).astype(float)

    cic_credit_utilization = np.clip(
        0.35 + 0.15 * latent_risk + rng.normal(0, 0.1, n_samples), 0, 1
    )

    cic_outstanding_balance_ratio = np.clip(
        0.40 + 0.20 * latent_risk + rng.normal(0, 0.12, n_samples), 0, 1
    )

    # Number of inquiries in last 6 months (Poisson, risk-dependent)
    cic_num_inquiries_6m = rng.poisson(
        lam=np.clip(1.5 + 1.2 * latent_risk, 0.1, 10), size=n_samples
    ).astype(float)

    # Account age: Gen Z thin-file → skewed toward 0–24 months
    cic_account_age_months = np.clip(
        rng.exponential(scale=14, size=n_samples) - 2 * latent_risk, 0, 120
    ).astype(float)

    cic_num_delinquencies = rng.poisson(
        lam=np.clip(0.3 + 0.6 * latent_risk, 0.01, 8), size=n_samples
    ).astype(float)

    cic_num_active_accounts = rng.poisson(
        lam=np.clip(2.0 - 0.4 * latent_risk, 0.5, 8), size=n_samples
    ).astype(float)

    # ── Behavioral Features ───────────────────────────────────────────────
    # Transaction frequency (negative correlation with risk)
    txn_frequency_30d = np.clip(
        rng.poisson(lam=np.clip(18 - 5 * latent_risk, 1, 50)), 0, 120
    ).astype(float)

    txn_frequency_60d = txn_frequency_30d * rng.uniform(1.6, 2.4, n_samples)
    txn_frequency_90d = txn_frequency_60d * rng.uniform(1.4, 1.8, n_samples)

    # Average transaction amount in VND (thousands)
    avg_txn_amount_30d = np.clip(
        rng.lognormal(mean=8.5 - 0.4 * latent_risk, sigma=0.8, size=n_samples), 50, 50_000
    )
    avg_txn_amount_90d = avg_txn_amount_30d * rng.uniform(0.85, 1.15, n_samples)

    # Merchant category diversity (1–12 categories)
    merchant_category_diversity = np.clip(
        rng.poisson(lam=np.clip(5 - 1.5 * latent_risk, 0.5, 12)), 1, 12
    ).astype(float)

    # Payment regularity: 0–1, higher = more regular
    digital_payment_regularity = np.clip(
        0.65 - 0.20 * latent_risk + rng.normal(0, 0.12, n_samples), 0, 1
    )

    # Mobile app engagement score (0–100)
    mobile_app_engagement_score = np.clip(
        60 - 15 * latent_risk + rng.normal(0, 10, n_samples), 0, 100
    )

    # Spending volatility (coefficient of variation), higher risk → more volatile
    spending_volatility_90d = np.clip(
        0.30 + 0.18 * latent_risk + rng.normal(0, 0.08, n_samples), 0, 1.5
    )

    # Salary credit regularity (binary)
    p_salary = np.clip(0.65 - 0.25 * latent_risk, 0.05, 0.95)
    salary_credit_regularity = rng.binomial(1, p_salary, n_samples).astype(float)

    # ── Demographic Features ──────────────────────────────────────────────
    age_at_application = np.clip(
        rng.integers(18, 27, size=n_samples) + rng.integers(0, 12, size=n_samples) / 12,
        18, 27
    )

    # 0=Student, 1=Informal, 2=Formal, 3=Self-employed
    employment_probs = [0.25, 0.30, 0.35, 0.10]
    employment_type_encoded = rng.choice(4, size=n_samples, p=employment_probs)

    # Income bracket 0–4 (inversely correlated with risk)
    income_raw = np.clip(2 - 0.8 * latent_risk + rng.normal(0, 0.5, n_samples), 0, 4)
    declared_income_bracket_encoded = np.floor(income_raw).clip(0, 4).astype(int)

    # Education: 0=High school, 1=College, 2=University, 3=Postgrad
    edu_probs = [0.10, 0.15, 0.65, 0.10]
    education_level_encoded = rng.choice(4, size=n_samples, p=edu_probs)

    # ── Default Flag (from latent risk + threshold calibration) ──────────
    threshold = norm.ppf(1 - default_rate)
    default_flag = (latent_risk > threshold).astype(int)

    actual_default_rate = default_flag.mean()
    logger.info(
        "Synthetic default rate: {:.2%} (target: {:.2%})",
        actual_default_rate, default_rate
    )

    # ── Assemble DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame({
        # Bureau
        "cic_payment_history_score":   cic_payment_history_score,
        "cic_credit_utilization":       cic_credit_utilization,
        "cic_outstanding_balance_ratio": cic_outstanding_balance_ratio,
        "cic_num_inquiries_6m":         cic_num_inquiries_6m,
        "cic_account_age_months":       cic_account_age_months,
        "cic_num_delinquencies":        cic_num_delinquencies,
        "cic_num_active_accounts":      cic_num_active_accounts,
        # Behavioral
        "txn_frequency_30d":            txn_frequency_30d,
        "txn_frequency_60d":            txn_frequency_60d,
        "txn_frequency_90d":            txn_frequency_90d,
        "avg_txn_amount_30d":           avg_txn_amount_30d,
        "avg_txn_amount_90d":           avg_txn_amount_90d,
        "merchant_category_diversity":  merchant_category_diversity,
        "digital_payment_regularity":   digital_payment_regularity,
        "mobile_app_engagement_score":  mobile_app_engagement_score,
        "spending_volatility_90d":      spending_volatility_90d,
        "salary_credit_regularity":     salary_credit_regularity,
        # Demographic
        "age_at_application":               age_at_application,
        "employment_type_encoded":          employment_type_encoded,
        "declared_income_bracket_encoded":  declared_income_bracket_encoded,
        "education_level_encoded":          education_level_encoded,
        # Target
        "default_flag": default_flag,
    })

    logger.success(
        "Synthetic dataset generated | shape={} | default_rate={:.2%}",
        df.shape, df["default_flag"].mean()
    )
    return df


def save_synthetic_data(df: pd.DataFrame, output_dir: str = "data/external") -> Path:
    """Persist the synthetic dataset to disk as CSV and Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        The synthetic dataset to save.
    output_dir : str
        Path relative to the project root.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    out = resolve_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "synthetic_credit_data.csv"
    parquet_path = out / "synthetic_credit_data.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    logger.success("Saved synthetic data | csv={} | parquet={}", csv_path, parquet_path)
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic credit scoring dataset"
    )
    parser.add_argument("--n_samples", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default_rate", type=float, default=0.175)
    parser.add_argument("--output_dir", type=str, default="data/external")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = generate_synthetic_dataset(
        n_samples=args.n_samples,
        random_seed=args.seed,
        default_rate=args.default_rate,
    )
    save_synthetic_data(df, output_dir=args.output_dir)
