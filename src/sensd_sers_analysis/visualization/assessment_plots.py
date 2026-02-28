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

from sensd_sers_analysis.assessment.model_consistency import (
    ConcentrationRegressionResult,
)


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


def plot_concentration_regression(
    df: pd.DataFrame,
    feature_col: str,
    *,
    regression_result: Optional[ConcentrationRegressionResult] = None,
    zero_cfu_baseline: Optional[float] = None,
    log_conc_col: str = "log_concentration",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Scatter plot of log concentration vs feature with regression line and 0 CFU baseline.

    Args:
        df: Feature DataFrame with log_concentration and feature columns.
        feature_col: Feature column (Y-axis).
        regression_result: Fitted regression (slope, intercept, R², RMSE).
        zero_cfu_baseline: Mean feature value for 0 CFU replicates (horizontal line).
        log_conc_col: Log concentration column (X-axis).
        title: Optional plot title.
        figsize: Figure size in inches.
        ax: Optional axes to draw on.

    Returns:
        matplotlib Figure.
    """
    if feature_col not in df.columns or log_conc_col not in df.columns:
        raise ValueError(
            f"Required columns '{feature_col}' or '{log_conc_col}' not in DataFrame."
        )

    valid = df[[log_conc_col, feature_col]].notna().all(axis=1)
    df_plot = df.loc[valid]

    if df_plot.empty:
        raise ValueError("No valid data for concentration regression plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x = df_plot[log_conc_col].astype(float).values
    y = df_plot[feature_col].astype(float).values

    ax.scatter(x, y, alpha=0.6, s=50, color="steelblue", edgecolors="white", zorder=3)

    # Regression line
    if regression_result is not None:
        x_min, x_max = x.min(), x.max()
        x_line = np.linspace(x_min, x_max, 50)
        y_line = regression_result.intercept + regression_result.slope * x_line
        ax.plot(
            x_line,
            y_line,
            color="crimson",
            linestyle="-",
            linewidth=2,
            label=f"Linear fit (R²={regression_result.r2:.3f}, RMSE={regression_result.rmse:.4f})",
            zorder=2,
        )

    # 0 CFU baseline
    if zero_cfu_baseline is not None:
        ax.axhline(
            zero_cfu_baseline,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label="0 CFU Baseline",
            zorder=1,
        )

    ax.set_xlabel("Log₁₀ Concentration (CFU/ml)")
    ax.set_ylabel(feature_col.replace("_", " ").title())
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    else:
        ax.set_title(
            f"Model-Based Consistency: {feature_col.replace('_', ' ').title()} vs Log Concentration",
            fontweight="bold",
            pad=12,
        )
    ax.legend(loc="best")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_multi_sensor_regression(
    df: pd.DataFrame,
    serotype: str,
    feature_col: str,
    *,
    sensor_col: str = "sensor_id",
    serotype_col: str = "serotype",
    log_conc_col: str = "log_concentration",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 7),
    line_alpha: float = 0.7,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Overlay scatter points and regression lines for all sensors (one serotype, one feature).

    Filters to the selected serotype and excludes 0 CFU (rows without valid
    log_concentration). Each sensor_id gets distinct color for points and line.
    Use to compare batch uniformity (tight bundle) vs erratic behavior (diverging).

    Args:
        df: Feature DataFrame with sensor_id, serotype, log_concentration, feature.
        serotype: Serotype to filter (e.g., "ST", "SE").
        feature_col: Feature column (Y-axis).
        sensor_col: Column for sensor identifier (hue).
        serotype_col: Column for serotype filter.
        log_conc_col: Log concentration column (X-axis).
        title: Optional plot title.
        figsize: Figure size in inches.
        line_alpha: Transparency for regression lines (0–1).
        ax: Optional axes to draw on.

    Returns:
        matplotlib Figure.
    """
    required = [sensor_col, serotype_col, log_conc_col, feature_col]
    if any(c not in df.columns for c in required):
        raise ValueError(f"Required columns missing. Need: {required}")

    # Filter to serotype and valid log_concentration (excludes 0 CFU)
    subset = df[
        (df[serotype_col].astype(str) == str(serotype))
        & df[log_conc_col].notna()
        & df[feature_col].notna()
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No valid data for serotype={serotype}. Need rows with non-null "
            f"{log_conc_col} and {feature_col}."
        )

    sensors = subset[sensor_col].dropna().unique()
    if len(sensors) == 0:
        raise ValueError(f"No sensor_id values found for serotype={serotype}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Distinct colors per sensor
    palette = sns.color_palette("husl", n_colors=len(sensors))
    color_map = dict(zip(sensors, palette))

    for sens in sensors:
        mask = subset[sensor_col] == sens
        sub = subset.loc[mask]
        x = sub[log_conc_col].astype(float).values
        y = sub[feature_col].astype(float).values
        color = color_map.get(sens, "gray")

        ax.scatter(
            x,
            y,
            alpha=0.6,
            s=50,
            color=color,
            edgecolors="white",
            label=str(sens),
            zorder=3,
        )

        if len(x) >= 2:
            res = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 50)
            y_line = res.intercept + res.slope * x_line
            ax.plot(
                x_line,
                y_line,
                color=color,
                linestyle="-",
                linewidth=2,
                alpha=line_alpha,
                zorder=2,
            )

    ax.set_xlabel("Log₁₀ Concentration (CFU/ml)")
    ax.set_ylabel(feature_col.replace("_", " ").title())
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    else:
        ax.set_title(
            f"Multi-Sensor Regression: {feature_col.replace('_', ' ').title()} "
            f"vs Log Concentration ({serotype})",
            fontweight="bold",
            pad=12,
        )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig
