"""
Exploratory Data Analysis Visualizations
==========================================
Publication-quality EDA plots for the Hybrid Credit Scoring dataset:
  - Feature distribution by default status
  - Default rate by feature bin (WoE diagnostic)
  - Correlation heatmap
  - Class imbalance chart
  - Behavioral vs. bureau feature IV comparison
  - Missing value heatmap

All functions save to outputs/figures/ and return the Axes object
for optional inline display in Jupyter notebooks.

Author : Phạm Tiến Dũng
Project: Hybrid Credit Scoring — NEU Vietnam
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import logger


# ─────────────────────────────────────────────────────────────
# Aesthetic configuration
# ─────────────────────────────────────────────────────────────

PALETTE_DEFAULT   = "#E74C3C"   # Red — default class
PALETTE_NDEFAULT  = "#2980B9"   # Blue — non-default class
PALETTE_BUREAU    = "#1ABC9C"   # Teal — bureau features
PALETTE_BEHAV     = "#9B59B6"   # Purple — behavioral features

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.color":        "#EEEEEE",
    "grid.linewidth":    0.8,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

FIGURES_DIR = "outputs/figures"


def _savefig(fig: plt.Figure, filename: str, dpi: int = 180) -> None:
    """Save figure to outputs/figures/."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Figure saved → {}", path)


# ─────────────────────────────────────────────────────────────
# 1. Class Imbalance Bar Chart
# ─────────────────────────────────────────────────────────────

def plot_class_distribution(
    y: pd.Series,
    save: bool = True,
    filename: str = "01_class_distribution.png",
) -> plt.Axes:
    """
    Bar chart showing the proportion of defaulted vs. non-defaulted loans.

    Parameters
    ----------
    y : pd.Series
        Binary target (1 = default, 0 = non-default).
    save : bool
    filename : str

    Returns
    -------
    matplotlib Axes
    """
    counts = y.value_counts().sort_index()
    labels = ["Non-Default (0)", "Default (1)"]
    colors = [PALETTE_NDEFAULT, PALETTE_DEFAULT]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, counts.values):
        pct = val / len(y) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + len(y) * 0.005,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title("Class Distribution — Default vs. Non-Default", fontweight="bold", pad=15)
    ax.set_ylabel("Number of Observations")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    if save:
        _savefig(fig, filename)
    return ax


# ─────────────────────────────────────────────────────────────
# 2. Feature Distribution by Default Status
# ─────────────────────────────────────────────────────────────

def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "default_flag",
    n_cols: int = 3,
    save: bool = True,
    filename: str = "02_feature_distributions.png",
) -> plt.Figure:
    """
    Grid of KDE plots for each feature, split by default status.

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str
    target_col : str
    n_cols : int
    save : bool
    filename : str

    Returns
    -------
    matplotlib Figure
    """
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feat in enumerate(features):
        ax = axes[i]
        for label, color, name in [
            (0, PALETTE_NDEFAULT, "Non-Default"),
            (1, PALETTE_DEFAULT,  "Default"),
        ]:
            subset = df[df[target_col] == label][feat].dropna()
            subset.plot.kde(ax=ax, label=name, color=color, linewidth=2)
        ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Default Status", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        _savefig(fig, filename)
    return fig


# ─────────────────────────────────────────────────────────────
# 3. Default Rate by Feature Bin
# ─────────────────────────────────────────────────────────────

def plot_default_rate_by_bin(
    series: pd.Series,
    target: pd.Series,
    n_bins: int = 10,
    feature_name: str = "",
    save: bool = True,
    filename: Optional[str] = None,
) -> plt.Axes:
    """
    Bar chart of observed default rate within each feature quintile.
    Useful as a WoE/IV diagnostic — monotone ordering indicates
    good feature discrimination.

    Parameters
    ----------
    series : pd.Series
    target : pd.Series
    n_bins : int
    feature_name : str
    save : bool
    filename : str or None

    Returns
    -------
    matplotlib Axes
    """
    df_tmp = pd.DataFrame({"feature": series, "target": target}).dropna()
    df_tmp["bin"] = pd.qcut(df_tmp["feature"], q=n_bins, duplicates="drop")
    rates = df_tmp.groupby("bin")["target"].mean() * 100
    counts = df_tmp.groupby("bin")["target"].count()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(rates)))
    bars = ax.bar(range(len(rates)), rates.values, color=colors, edgecolor="white")

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"n={count:,}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([str(b) for b in rates.index], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Default Rate (%)")
    ax.set_title(f"Default Rate by Bin — {feature_name or series.name}", fontweight="bold")

    plt.tight_layout()
    fname = filename or f"default_rate_{feature_name or 'feature'}.png"
    if save:
        _savefig(fig, fname)
    return ax


# ─────────────────────────────────────────────────────────────
# 4. Correlation Heatmap
# ─────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    save: bool = True,
    filename: str = "03_correlation_heatmap.png",
) -> plt.Axes:
    """
    Heatmap of Pearson correlations between features.

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str or None (uses all numeric columns)
    save : bool
    filename : str

    Returns
    -------
    matplotlib Axes
    """
    cols = features or df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[cols].corr(method="pearson")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 0.7), max(8, len(cols) * 0.6)))

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.5,
        annot_kws={"size": 8}, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix (Pearson)", fontweight="bold", pad=15)
    plt.tight_layout()
    if save:
        _savefig(fig, filename)
    return ax


# ─────────────────────────────────────────────────────────────
# 5. IV Bar Chart — Bureau vs. Behavioral
# ─────────────────────────────────────────────────────────────

def plot_iv_comparison(
    iv_summary: pd.DataFrame,
    bureau_features: List[str],
    behavioral_features: List[str],
    save: bool = True,
    filename: str = "04_iv_comparison.png",
) -> plt.Axes:
    """
    Horizontal bar chart of IV values, colored by feature category.

    Parameters
    ----------
    iv_summary : pd.DataFrame
        Output of IVFeatureSelector.report() — columns: feature, iv, status
    bureau_features : list of str
    behavioral_features : list of str
    save : bool
    filename : str

    Returns
    -------
    matplotlib Axes
    """
    df = iv_summary.copy().sort_values("iv", ascending=True)
    colors = [
        PALETTE_BUREAU if f in bureau_features else
        PALETTE_BEHAV  if f in behavioral_features else
        "#BDC3C7"
        for f in df["feature"]
    ]

    fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.45)))
    bars = ax.barh(df["feature"], df["iv"], color=colors, edgecolor="white", height=0.7)

    ax.axvline(x=0.02, color="gray", linestyle="--", linewidth=1, label="IV=0.02 (min threshold)")
    ax.axvline(x=0.50, color="red",  linestyle=":",  linewidth=1, label="IV=0.50 (leakage warning)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE_BUREAU, label="CIC Bureau"),
        Patch(facecolor=PALETTE_BEHAV,  label="Behavioral"),
        plt.Line2D([0], [0], color="gray", linestyle="--", label="IV=0.02"),
        plt.Line2D([0], [0], color="red",  linestyle=":",  label="IV=0.50"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_xlabel("Information Value (IV)")
    ax.set_title("Feature Information Value — Bureau vs. Behavioral", fontweight="bold")
    plt.tight_layout()
    if save:
        _savefig(fig, filename)
    return ax


# ─────────────────────────────────────────────────────────────
# 6. Missing Value Heatmap
# ─────────────────────────────────────────────────────────────

def plot_missing_heatmap(
    df: pd.DataFrame,
    save: bool = True,
    filename: str = "05_missing_values.png",
) -> plt.Axes:
    """
    Heatmap of missing value patterns across features.

    Parameters
    ----------
    df : pd.DataFrame
    save : bool
    filename : str

    Returns
    -------
    matplotlib Axes
    """
    missing = df.isnull()
    pct_missing = (missing.mean() * 100).sort_values(ascending=False)
    pct_missing = pct_missing[pct_missing > 0]

    if pct_missing.empty:
        logger.info("No missing values found — skipping missing value heatmap")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return ax

    fig, ax = plt.subplots(figsize=(10, max(4, len(pct_missing) * 0.4)))
    colors = ["#E74C3C" if v > 30 else "#F39C12" if v > 10 else "#2980B9" for v in pct_missing.values]
    ax.barh(pct_missing.index, pct_missing.values, color=colors, edgecolor="white")
    ax.set_xlabel("Missing Values (%)")
    ax.set_title("Missing Value Analysis by Feature", fontweight="bold")
    ax.axvline(x=20, color="red", linestyle="--", linewidth=1, label="20% threshold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        _savefig(fig, filename)
    return ax
