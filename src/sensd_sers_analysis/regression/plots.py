"""
Visualization helpers for concentration regression paradigms.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sensd_sers_analysis.regression.models_global import SingleRegressorResult


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Actual vs predicted (log10 concentration)",
    axis_label: str = "log10 concentration",
    hue: Optional[np.ndarray] = None,
    hue_name: str = "Serotype",
    figsize: tuple[float, float] = (7, 6),
) -> plt.Figure:
    """
    Scatter of held-out truth vs predictions with y = x reference.

    Parameters
    ----------
    y_true, y_pred:
        1-D arrays of equal length.
    title:
        Figure title.
    axis_label:
        Label for both axes.
    hue:
        Optional categorical labels per point (e.g. serotype strings).
    hue_name:
        Legend title when ``hue`` is provided.
    figsize:
        Matplotlib figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (caller should close if needed).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    fig, ax = plt.subplots(figsize=figsize)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1, label="Ideal (y = x)")
    if hue is not None:
        plot_df = pd.DataFrame(
            {
                "Actual": y_true,
                "Predicted": y_pred,
                hue_name: hue.astype(str),
            }
        )
        sns.scatterplot(
            data=plot_df,
            x="Actual",
            y="Predicted",
            hue=hue_name,
            alpha=0.75,
            s=55,
            ax=ax,
        )
    else:
        ax.scatter(y_true, y_pred, alpha=0.75, s=55, edgecolors="k", linewidths=0.3)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Residuals vs predicted (log10)",
    figsize: tuple[float, float] = (7, 4.5),
) -> plt.Figure:
    """
    Residual (actual − predicted) vs predicted values.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    resid = y_true - y_pred
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.scatter(y_pred, resid, alpha=0.75, s=45, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Predicted log10 concentration")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_regression_feature_importance(
    result: SingleRegressorResult,
    *,
    top_k: int = 15,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Horizontal bar chart of Random Forest feature importances (global model).
    """
    est = result.model
    if not hasattr(est, "feature_importances_"):
        raise ValueError("Model has no feature_importances_; use a tree-based regressor.")
    imp = np.asarray(est.feature_importances_, dtype=np.float64)
    names = list(result.feature_names)
    order = np.argsort(imp)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(order))
    ax.barh(y_pos, imp[order], align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature importance — {result.model_name}", fontweight="bold", pad=10)
    fig.tight_layout()
    return fig
