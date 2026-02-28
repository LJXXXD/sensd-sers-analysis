"""
Visualization for sensor assessment: degradation trends and batch comparisons.

Provides publication-ready plots for the Sensor Assessment & Report module.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def plot_degradation_trend(
    df: pd.DataFrame,
    feature_col: str,
    sequence_col: str,
    *,
    group_col: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot feature vs. sequence with linear trendline for degradation analysis.

    A negative slope indicates degradation. Points are plotted with the
    fitted line overlayed.

    Args:
        df: Feature DataFrame with feature_col and sequence_col.
        feature_col: Y-axis (e.g., max_intensity, integral_area).
        sequence_col: X-axis (e.g., signal_index, sequence).
        group_col: If provided, plot one subplot per group (e.g., sensor_id).
        title: Optional plot title.
        figsize: Figure size in inches.
        ax: Optional axes to draw on.

    Returns:
        matplotlib Figure.
    """
    if feature_col not in df.columns or sequence_col not in df.columns:
        raise ValueError(
            f"Required columns '{feature_col}' or '{sequence_col}' not in DataFrame."
        )

    df_clean = df.dropna(subset=[feature_col, sequence_col])
    if df_clean.empty:
        raise ValueError("No valid data for degradation plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if group_col and group_col in df.columns:
        groups = df_clean[group_col].dropna().unique()
        if len(groups) > 1:
            # Multiple groups: use hue
            for g in groups:
                mask = df_clean[group_col] == g
                x = df_clean.loc[mask, sequence_col].astype(float).values
                y = df_clean.loc[mask, feature_col].astype(float).values
                ax.scatter(x, y, alpha=0.6, s=40, label=str(g))
                if len(x) >= 2:
                    res = stats.linregress(x, y)
                    x_line = np.linspace(x.min(), x.max(), 50)
                    ax.plot(
                        x_line,
                        res.intercept + res.slope * x_line,
                        alpha=0.8,
                        linestyle="--",
                    )
            ax.legend(loc="best", fontsize=8)
        else:
            group_col = None  # Single group, fall through

    if group_col is None or group_col not in df.columns:
        x = df_clean[sequence_col].astype(float).values
        y = df_clean[feature_col].astype(float).values
        ax.scatter(x, y, alpha=0.6, s=50, color="steelblue", edgecolors="white")

        if len(x) >= 2:
            res = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(
                x_line,
                res.intercept + res.slope * x_line,
                color="crimson",
                linestyle="--",
                linewidth=2,
                label=f"Slope={res.slope:.4f} (R²={res.rvalue**2:.3f})",
            )
            ax.legend(loc="best")

    ax.set_xlabel(sequence_col.replace("_", " ").title())
    ax.set_ylabel(feature_col.replace("_", " ").title())
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    else:
        ax.set_title(
            f"Degradation Trend: {feature_col.replace('_', ' ').title()} vs {sequence_col.replace('_', ' ').title()}",
            fontweight="bold",
            pad=12,
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_batch_boxplot(
    df: pd.DataFrame,
    feature_col: str,
    *,
    sensor_col: str = "sensor_id",
    group_col: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Side-by-side boxplots of feature by sensor for batch variance comparison.

    Quickly identifies sensors deviating from the population distribution.

    Args:
        df: Feature DataFrame.
        feature_col: Feature to plot.
        sensor_col: Column for sensor identification (x-axis).
        group_col: Optional hue for stratification (e.g., concentration_group).
        title: Optional plot title.
        figsize: Figure size.
        ax: Optional axes.

    Returns:
        matplotlib Figure.
    """
    if feature_col not in df.columns or sensor_col not in df.columns:
        raise ValueError(
            f"Required columns '{feature_col}' or '{sensor_col}' not in DataFrame."
        )

    df_clean = df.dropna(subset=[feature_col])
    if df_clean.empty:
        raise ValueError("No valid data for batch boxplot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    hue = group_col if group_col and group_col in df.columns else None
    sns.boxplot(
        data=df_clean,
        x=sensor_col,
        y=feature_col,
        hue=hue,
        ax=ax,
        palette="muted" if hue else None,
    )
    ax.tick_params(
        axis="x", rotation=45 if len(df_clean[sensor_col].unique()) > 5 else 0
    )
    ax.set_xlabel(sensor_col.replace("_", " ").title())
    ax.set_ylabel(feature_col.replace("_", " ").title())
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    else:
        ax.set_title(
            f"Batch Variance: {feature_col.replace('_', ' ').title()} by {sensor_col.replace('_', ' ').title()}",
            fontweight="bold",
            pad=12,
        )

    leg = ax.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1.02, 1))
        leg.set_loc("upper left")

    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
