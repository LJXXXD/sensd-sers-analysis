"""
Phase 2 classification visualizations.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .models import ClassificationResult


def plot_pca_classification(
    df: pd.DataFrame,
    *,
    pc1_col: str = "PC1",
    pc2_col: str = "PC2",
    target_col: str = "target",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (9, 7),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    PCA scatter colored by target (ST, SE, Rinsate).

    Args:
        df: DataFrame with PC1, PC2, and target.
        pc1_col: PC1 column name.
        pc2_col: PC2 column name.
        target_col: Target column.
        title: Optional plot title.
        figsize: Figure size.
        ax: Optional axes.

    Returns:
        matplotlib Figure.
    """
    if pc1_col not in df.columns or pc2_col not in df.columns:
        raise ValueError(f"Need {pc1_col} and {pc2_col} in DataFrame")
    if target_col not in df.columns:
        raise ValueError(f"Need {target_col} in DataFrame")

    subset = df[[pc1_col, pc2_col, target_col]].dropna()
    if subset.empty:
        raise ValueError("No valid PCA + target data")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    hue_order = ["ST", "SE", "Rinsate"]
    present = [c for c in hue_order if c in subset[target_col].unique()]
    if not present:
        present = sorted(subset[target_col].unique().tolist())

    sns.scatterplot(
        data=subset,
        x=pc1_col,
        y=pc2_col,
        hue=target_col,
        hue_order=present,
        alpha=0.7,
        s=60,
        ax=ax,
        legend="brief",
    )
    ax.set_xlabel(pc1_col)
    ax.set_ylabel(pc2_col)
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    else:
        ax.set_title(
            "PCA Scatter: PC1 vs PC2 by Class",
            fontweight="bold",
            pad=12,
        )
    ax.legend(loc="best", framealpha=0.9)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    result: ClassificationResult,
    *,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (6, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Heatmap-style confusion matrix.

    Args:
        result: ClassificationResult with confusion_matrix and class_names.
        title: Optional title.
        figsize: Figure size.
        ax: Optional axes.

    Returns:
        matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    labels = result.class_names
    cm = result.confusion_matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count"},
        square=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    else:
        ax.set_title(
            f"{result.model_name} Confusion Matrix",
            fontweight="bold",
            pad=12,
        )
    fig.tight_layout()
    return fig


def plot_feature_importance(
    result: ClassificationResult,
    *,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of Random Forest feature importances.

    Args:
        result: ClassificationResult with feature_importances.
        top_n: Show top N features (default: all).
        title: Optional title.
        figsize: Figure size.
        ax: Optional axes.

    Returns:
        matplotlib Figure.
    """
    if result.feature_importances is None:
        raise ValueError("ClassificationResult has no feature_importances")

    imp = result.feature_importances
    names = result.feature_names
    order = np.argsort(imp)[::-1]
    if top_n is not None:
        order = order[:top_n]
    imp_sorted = imp[order]
    names_sorted = [names[i] for i in order]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_pos = np.arange(len(names_sorted))
    ax.barh(y_pos, imp_sorted, color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.replace("_", " ").title() for n in names_sorted])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    else:
        ax.set_title(
            f"{result.model_name} Feature Importance",
            fontweight="bold",
            pad=12,
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
