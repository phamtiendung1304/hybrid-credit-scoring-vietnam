"""
SHAP Interpretability Visualizations
=======================================
Publication-quality SHAP plots for post-hoc model explanation.
Implements the three core visualization types used in the research report:

  1. SHAP Summary Plot     — global feature importance (beeswarm)
  2. Feature Attribution   — bureau vs. behavioral contribution pie/bar
  3. SHAP Dependence Plot  — marginal effect of top features
  4. Waterfall Plot        — individual-level local explanation

SHAP (SHapley Additive exPlanations) provides theoretically grounded
feature attributions satisfying efficiency, symmetry, dummy, and
linearity axioms (Lundberg & Lee, 2017).

Reference:
    Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to
    Interpreting Model Predictions. NeurIPS 2017.

Author : Phạm Tiến Dũng
Project: Hybrid Credit Scoring — NEU Vietnam
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap

from src.utils.logger import logger


FIGURES_DIR = "outputs/figures"

PALETTE_BUREAU = "#1ABC9C"
PALETTE_BEHAV  = "#9B59B6"
PALETTE_DEMO   = "#F39C12"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.size":        11,
    "axes.titlesize":   13,
})


def _savefig(fig: plt.Figure, filename: str, dpi: int = 180) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("SHAP figure saved → {}", path)


# ─────────────────────────────────────────────────────────────
# 1. SHAP Summary Plot (beeswarm)
# ─────────────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_features: int = 20,
    save: bool = True,
    filename: str = "07_shap_summary_beeswarm.png",
) -> None:
    """
    Beeswarm SHAP summary plot showing global feature importance and
    directional effect of each feature on the default probability.

    Red points = high feature value, Blue = low.
    X-axis position = SHAP value (impact on log-odds of default).

    Parameters
    ----------
    shap_values : np.ndarray, shape (n_samples, n_features)
    X : pd.DataFrame
        Feature matrix (same columns as shap_values).
    max_features : int
        Number of top features to show.
    save : bool
    filename : str
    """
    logger.info("Generating SHAP beeswarm summary plot")
    fig, ax = plt.subplots(figsize=(10, max(6, max_features * 0.38)))

    shap.summary_plot(
        shap_values, X, max_display=max_features,
        show=False, plot_type="dot",
    )
    plt.title("SHAP Feature Importance — Hybrid Credit Scoring Model",
              fontweight="bold", fontsize=13, pad=15)
    if save:
        _savefig(plt.gcf(), filename)
    plt.close()


# ─────────────────────────────────────────────────────────────
# 2. Feature Attribution: Bureau vs. Behavioral Bar Chart
# ─────────────────────────────────────────────────────────────

def plot_feature_attribution(
    shap_values: np.ndarray,
    feature_names: List[str],
    bureau_features: List[str],
    behavioral_features: List[str],
    demographic_features: Optional[List[str]] = None,
    save: bool = True,
    filename: str = "08_shap_feature_attribution.png",
) -> pd.DataFrame:
    """
    Grouped bar chart comparing total SHAP contribution from bureau,
    behavioral, and demographic feature groups.

    Also returns the attribution breakdown as a DataFrame.

    Parameters
    ----------
    shap_values : np.ndarray
    feature_names : list of str
    bureau_features : list of str
    behavioral_features : list of str
    demographic_features : list of str or None
    save : bool
    filename : str

    Returns
    -------
    pd.DataFrame with columns: group, total_mean_abs_shap, pct_contribution
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    total = mean_abs.sum()

    groups: Dict[str, List[str]] = {
        "CIC Bureau":  bureau_features,
        "Behavioral":  behavioral_features,
    }
    if demographic_features:
        groups["Demographic"] = demographic_features

    records = []
    for group, features in groups.items():
        indices = [i for i, f in enumerate(feature_names) if f in features]
        group_total = mean_abs[indices].sum() if indices else 0.0
        records.append({
            "group":               group,
            "total_mean_abs_shap": round(float(group_total), 4),
            "pct_contribution":    round(float(group_total / total * 100), 2),
        })

    df_attr = pd.DataFrame(records).sort_values("pct_contribution", ascending=False)

    # Plot
    colors = [PALETTE_BUREAU, PALETTE_BEHAV, PALETTE_DEMO][:len(df_attr)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart — absolute SHAP contribution
    bars = ax1.barh(df_attr["group"], df_attr["total_mean_abs_shap"],
                    color=colors, edgecolor="white", height=0.5)
    for bar, row in zip(bars, df_attr.itertuples()):
        ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{row.pct_contribution:.1f}%",
                 va="center", ha="left", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Total Mean |SHAP| Value")
    ax1.set_title("SHAP Attribution by Feature Group", fontweight="bold")

    # Pie chart
    wedges, _, autotexts = ax2.pie(
        df_attr["pct_contribution"],
        labels=df_attr["group"],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.7,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
    ax2.set_title("Proportional Attribution", fontweight="bold")

    fig.suptitle(
        "Hybrid Credit Model — Feature Group Attribution (SHAP)\n"
        f"Behavioral data contributes {df_attr.loc[df_attr.group=='Behavioral','pct_contribution'].values[0]:.1f}% "
        "of total predictive power",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if save:
        _savefig(fig, filename)
    plt.close()
    return df_attr


# ─────────────────────────────────────────────────────────────
# 3. SHAP Dependence Plot — Top Feature
# ─────────────────────────────────────────────────────────────

def plot_shap_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = "auto",
    save: bool = True,
    filename: Optional[str] = None,
) -> None:
    """
    Scatter plot of feature values vs. their SHAP values,
    colored by an interaction feature.

    Reveals the marginal effect of `feature` on default probability
    and potential interaction effects.

    Parameters
    ----------
    shap_values : np.ndarray
    X : pd.DataFrame
    feature : str
        Primary feature to plot.
    interaction_feature : str
        Feature to color points by. 'auto' selects the highest
        SHAP interaction.
    save : bool
    filename : str or None
    """
    logger.info("Generating SHAP dependence plot for '{}'", feature)
    shap.dependence_plot(
        feature, shap_values, X,
        interaction_index=interaction_feature,
        show=False,
    )
    plt.title(f"SHAP Dependence — {feature}", fontweight="bold")
    fname = filename or f"shap_dependence_{feature}.png"
    if save:
        _savefig(plt.gcf(), fname)
    plt.close()


# ─────────────────────────────────────────────────────────────
# 4. Waterfall Plot — Individual Explanation
# ─────────────────────────────────────────────────────────────

def plot_waterfall_explanation(
    explainer: shap.Explainer,
    X_single: pd.DataFrame,
    observation_idx: int = 0,
    save: bool = True,
    filename: str = "10_shap_waterfall_example.png",
) -> None:
    """
    Waterfall plot explaining a single prediction.

    Shows how each feature pushed the model output above or below
    the expected value (E[f(X)]).

    Parameters
    ----------
    explainer : shap.Explainer
        Fitted SHAP explainer.
    X_single : pd.DataFrame
        Single observation (1 row) or full dataset (indexed).
    observation_idx : int
        Row index to explain.
    save : bool
    filename : str
    """
    logger.info("Generating SHAP waterfall for observation idx={}", observation_idx)
    shap_vals = explainer(X_single.iloc[[observation_idx]])
    shap.waterfall_plot(shap_vals[0], max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Observation #{observation_idx}", fontweight="bold", pad=15)
    if save:
        _savefig(plt.gcf(), filename)
    plt.close()


# ─────────────────────────────────────────────────────────────
# 5. Top N Feature Importance Bar (mean |SHAP|)
# ─────────────────────────────────────────────────────────────

def plot_shap_bar_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    bureau_features: List[str],
    behavioral_features: List[str],
    top_n: int = 15,
    save: bool = True,
    filename: str = "09_shap_bar_importance.png",
) -> None:
    """
    Horizontal bar chart of mean absolute SHAP values with
    bureau/behavioral color coding.

    Parameters
    ----------
    shap_values : np.ndarray
    feature_names : list of str
    bureau_features : list of str
    behavioral_features : list of str
    top_n : int
    save : bool
    filename : str
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df = df.nlargest(top_n, "mean_abs_shap").sort_values("mean_abs_shap", ascending=True)

    colors = [
        PALETTE_BUREAU if f in bureau_features else
        PALETTE_BEHAV  if f in behavioral_features else
        PALETTE_DEMO
        for f in df["feature"]
    ]

    fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.45)))
    ax.barh(df["feature"], df["mean_abs_shap"], color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Mean |SHAP Value|  (average impact on default probability)")
    ax.set_title(f"Top {top_n} Features — Global SHAP Importance", fontweight="bold")

    legend_elements = [
        mpatches.Patch(facecolor=PALETTE_BUREAU, label="CIC Bureau"),
        mpatches.Patch(facecolor=PALETTE_BEHAV,  label="Behavioral"),
        mpatches.Patch(facecolor=PALETTE_DEMO,   label="Demographic"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    if save:
        _savefig(fig, filename)
    plt.close()
