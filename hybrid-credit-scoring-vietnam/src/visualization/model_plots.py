"""
Model Performance Visualizations
==================================
Publication-quality plots for credit model evaluation:
  • ROC Curve with AUC annotation
  • Precision-Recall Curve
  • KS Separation Plot (Cumulative Score Distribution)
  • Score Distribution by Default Status
  • Model Comparison Bar Chart
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

from src.utils.config import CONFIG
from src.utils.logger import logger


_STYLE      = CONFIG["visualization"]["style"]
_DPI        = CONFIG["visualization"]["figure_dpi"]
_FMT        = CONFIG["visualization"]["figure_format"]
_FIGSIZE    = tuple(CONFIG["visualization"]["figsize_default"])
_FIGSIZE_W  = tuple(CONFIG["visualization"]["figsize_wide"])

plt.style.use(_STYLE)


# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    results: list[dict],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot overlaid ROC curves for multiple models.

    Parameters
    ----------
    results : list[dict]
        Each dict must have keys: ``'name'``, ``'y_true'``, ``'y_score'``.
    save_path : str | Path | None
        If provided, saves the figure to this path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    colors = sns.color_palette("Blues_d", n_colors=len(results) + 2)[2:]

    for res, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_score"])
        auc = roc_auc_score(res["y_true"], res["y_score"])
        gini = 2 * auc - 1
        ax.plot(
            fpr, tpr,
            label=f"{res['name']}  (AUC = {auc:.4f} | Gini = {gini:.4f})",
            color=color,
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Hybrid Credit Scoring Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, save_path)
    return fig


def plot_ks_separation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot KS separation between default and non-default score distributions.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Predicted PD scores.
    model_name : str
        Title label.
    save_path : str | Path | None

    Returns
    -------
    plt.Figure
    """
    scores_default     = np.sort(y_score[y_true == 1])
    scores_non_default = np.sort(y_score[y_true == 0])

    cdf_default     = np.arange(1, len(scores_default) + 1)     / len(scores_default)
    cdf_non_default = np.arange(1, len(scores_non_default) + 1) / len(scores_non_default)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(scores_default,     cdf_default,     color="#D62728", label="Default (y=1)", linewidth=2)
    ax.plot(scores_non_default, cdf_non_default, color="#1F77B4", label="Non-Default (y=0)", linewidth=2)

    # Annotate KS statistic
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks_idx = np.argmax(tpr - fpr)
    ks_value = tpr[ks_idx] - fpr[ks_idx]
    ks_threshold = thresholds[ks_idx]

    ax.axvline(ks_threshold, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.annotate(
        f"KS = {ks_value:.4f}",
        xy=(ks_threshold, 0.5),
        fontsize=11, color="black",
        ha="left",
        xytext=(ks_threshold + 0.02, 0.5),
    )

    ax.set_xlabel("Predicted PD Score", fontsize=12)
    ax.set_ylabel("Cumulative Distribution", fontsize=12)
    ax.set_title(f"KS Separation — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, save_path)
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Bar chart comparing AUC and Gini across models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from ``src.evaluation.metrics.compare_models``.
        Must contain columns: ``'model'``, ``'auc'``, ``'gini'``.
    save_path : str | Path | None

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=_FIGSIZE_W)
    palette = sns.color_palette("Blues_d", n_colors=len(comparison_df) + 2)[2:]

    for ax, metric, label in zip(
        axes,
        ["auc", "gini"],
        ["AUC-ROC", "Gini Coefficient"],
    ):
        bars = ax.barh(
            comparison_df["model"],
            comparison_df[metric],
            color=palette,
            edgecolor="white",
            height=0.5,
        )
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
        ax.set_xlabel(label, fontsize=12)
        ax.set_title(f"{label} Comparison", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Model Performance Comparison — Hybrid Credit Scoring", fontsize=14, fontweight="bold")
    fig.tight_layout()

    _save_figure(fig, save_path)
    return fig


def plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Histogram of predicted PD scores by default status.

    Parameters
    ----------
    y_true : np.ndarray
    y_score : np.ndarray
    model_name : str
    save_path : str | Path | None

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(
        y_score[y_true == 0], bins=50, alpha=0.65,
        color="#1F77B4", label="Non-Default (y=0)", density=True
    )
    ax.hist(
        y_score[y_true == 1], bins=50, alpha=0.65,
        color="#D62728", label="Default (y=1)", density=True
    )
    ax.set_xlabel("Predicted PD Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Score Distribution by Default Status — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────

def _save_figure(fig: plt.Figure, path: Optional[str | Path]) -> None:
    """Save figure to disk if a path is provided."""
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight", format=_FMT)
        logger.success("Figure saved to {}", path)
    plt.close(fig)
