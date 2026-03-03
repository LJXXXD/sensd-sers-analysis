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
    MacroRegressionResult,
    compute_macro_batch_regression,
    fit_concentration_regression_cleaned,
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
    raw_regression_result: Optional[ConcentrationRegressionResult] = None,
    zero_cfu_baseline: Optional[float] = None,
    outlier_mask: Optional[np.ndarray] = None,
    log_conc_col: str = "log_concentration",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Scatter plot of log concentration vs feature with dual regression lines.

    When raw_regression_result is provided (e.g., from cleaned fit), draws both:
    - Raw fit: dashed gray (before outlier removal).
    - Clean fit: solid red (after outlier removal).
    This justifies outlier removal by showing before-and-after.

    Args:
        df: Feature DataFrame with log_concentration and feature columns.
        feature_col: Feature column (Y-axis).
        regression_result: Clean fitted regression (prominent solid red line).
        raw_regression_result: Raw fitted regression (dashed gray, before outliers removed).
        zero_cfu_baseline: Mean feature value for 0 CFU replicates (horizontal line).
        outlier_mask: Optional boolean array (True=outlier). Outliers drawn as red X.
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

    # Plot inliers first, then outliers with distinct marker
    if outlier_mask is not None and len(outlier_mask) == len(x):
        inlier_mask = ~outlier_mask
        ax.scatter(
            x[inlier_mask],
            y[inlier_mask],
            alpha=0.6,
            s=50,
            color="steelblue",
            edgecolors="white",
            zorder=3,
        )
        if np.any(outlier_mask):
            ax.scatter(
                x[outlier_mask],
                y[outlier_mask],
                marker="x",
                s=80,
                color="red",
                linewidths=2,
                label="Outlier (excluded from fit)",
                zorder=4,
            )
    else:
        ax.scatter(
            x, y, alpha=0.6, s=50, color="steelblue", edgecolors="white", zorder=3
        )

    x_min, x_max = x.min(), x.max()
    x_line = np.linspace(x_min, x_max, 50)

    # Line 1: Raw fit (dashed gray) — before outlier removal
    if raw_regression_result is not None:
        y_raw = raw_regression_result.intercept + raw_regression_result.slope * x_line
        ax.plot(
            x_line,
            y_raw,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"Raw fit (R²={raw_regression_result.r2:.3f}, RMSE={raw_regression_result.rmse:.4f})",
            zorder=2,
        )

    # Line 2: Clean fit (solid red) — after outlier removal
    if regression_result is not None:
        y_clean = regression_result.intercept + regression_result.slope * x_line
        ax.plot(
            x_line,
            y_clean,
            color="crimson",
            linestyle="-",
            linewidth=2,
            label=f"Clean fit (R²={regression_result.r2:.3f}, RMSE={regression_result.rmse:.4f})",
            zorder=3,
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
    excluded_sensors: Optional[set[str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 7),
    line_alpha: float = 0.7,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Overlay scatter points and regression lines for all sensors (one serotype, one feature).

    Filters to the selected serotype and excludes 0 CFU (rows without valid
    log_concentration). Excluded sensors (from batch QA) are drawn with dashed
    gray lines; passing sensors use distinct bright colors.

    Args:
        df: Feature DataFrame with sensor_id, serotype, log_concentration, feature.
        serotype: Serotype to filter (e.g., "ST", "SE").
        feature_col: Feature column (Y-axis).
        sensor_col: Column for sensor identifier (hue).
        serotype_col: Column for serotype filter.
        log_conc_col: Log concentration column (X-axis).
        excluded_sensors: Sensor IDs marked Excluded in batch QA (dashed/gray).
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

    excluded = excluded_sensors or set()
    excluded = {str(s) for s in excluded}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Distinct colors for passing sensors; gray for excluded
    passing_sensors = [s for s in sensors if str(s) not in excluded]
    palette = sns.color_palette("husl", n_colors=max(len(passing_sensors), 1))
    color_map = dict(zip(passing_sensors, palette))

    for sens in sensors:
        sens_str = str(sens)
        is_excluded = sens_str in excluded
        color = "gray" if is_excluded else color_map.get(sens, "gray")
        label = f"{sens_str} (Excluded)" if is_excluded else str(sens_str)

        mask = subset[sensor_col] == sens
        sub = subset.loc[mask]
        x = sub[log_conc_col].astype(float).values
        y = sub[feature_col].astype(float).values

        ax.scatter(
            x,
            y,
            alpha=0.4 if is_excluded else 0.6,
            s=40 if is_excluded else 50,
            color=color,
            edgecolors="white",
            label=label,
            zorder=2 if is_excluded else 3,
        )

        if len(x) >= 2:
            res = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 50)
            y_line = res.intercept + res.slope * x_line
            ax.plot(
                x_line,
                y_line,
                color=color,
                linestyle="--" if is_excluded else "-",
                linewidth=1.5 if is_excluded else 2,
                alpha=0.5 if is_excluded else line_alpha,
                zorder=1 if is_excluded else 2,
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
    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.85,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_macro_batch_regression(
    df: pd.DataFrame,
    serotype: str,
    feature_col: str,
    pass_sensors: set[str],
    *,
    sensor_col: str = "sensor_id",
    serotype_col: str = "serotype",
    log_conc_col: str = "log_concentration",
    macro_result: Optional[MacroRegressionResult] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, Optional[MacroRegressionResult]]:
    """
    Pool inlier data from Pass sensors and plot macro-regression line.

    Aggregates all valid inlier points from sensors that passed QA, fits a
    single macro-regression, and displays Batch RMSE and Batch R². Use this
    to assess overall batch consistency.

    Args:
        df: Feature DataFrame.
        serotype: Serotype to filter.
        feature_col: Feature column (Y-axis).
        pass_sensors: Sensor IDs that passed QA.
        sensor_col: Sensor identifier column.
        serotype_col: Serotype column.
        log_conc_col: Log concentration column (X-axis).
        macro_result: Pre-computed result; if None, computed internally.
        title: Optional plot title.
        figsize: Figure size in inches.
        ax: Optional axes to draw on.

    Returns:
        Tuple of (Figure, MacroRegressionResult or None if insufficient data).
    """
    required = [sensor_col, serotype_col, log_conc_col, feature_col]
    if any(c not in df.columns for c in required):
        raise ValueError(f"Required columns missing. Need: {required}")

    subset = df[
        (df[serotype_col].astype(str) == str(serotype))
        & df[log_conc_col].notna()
        & df[feature_col].notna()
    ]
    if subset.empty:
        raise ValueError(
            f"No valid data for serotype={serotype}. Need non-null "
            f"{log_conc_col} and {feature_col}."
        )

    # Pool inlier data from pass sensors
    x_pooled: list[float] = []
    y_pooled: list[float] = []
    sensor_pooled: list[str] = []

    for sens in pass_sensors:
        sub = subset[subset[sensor_col].astype(str) == str(sens)]
        if sub.empty:
            continue
        cres = fit_concentration_regression_cleaned(
            sub, feature_col, log_conc_col=log_conc_col
        )
        if cres is None:
            continue
        valid = sub[[log_conc_col, feature_col]].notna().all(axis=1)
        sub_fit = sub.loc[valid]
        x_vals = sub_fit[log_conc_col].astype(float).values
        y_vals = sub_fit[feature_col].astype(float).values
        inlier_mask = ~cres.outlier_mask
        x_pooled.extend(x_vals[inlier_mask].tolist())
        y_pooled.extend(y_vals[inlier_mask].tolist())
        sensor_pooled.extend([str(sens)] * int(np.sum(inlier_mask)))

    if len(x_pooled) < 2:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.text(
            0.5,
            0.5,
            "Insufficient pooled data from Pass sensors (need ≥2 points)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig, None

    x_arr = np.array(x_pooled)
    y_arr = np.array(y_pooled)
    result = macro_result or compute_macro_batch_regression(
        df,
        serotype,
        feature_col,
        pass_sensors,
        sensor_col=sensor_col,
        serotype_col=serotype_col,
        log_conc_col=log_conc_col,
    )

    if result is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.text(
            0.5,
            0.5,
            "Insufficient pooled data for macro regression",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig, None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Separate inliers vs macro outliers
    inlier_mask = ~result.macro_outlier_mask
    x_in = x_arr[inlier_mask]
    y_in = y_arr[inlier_mask]
    sensor_in = [s for s, m in zip(sensor_pooled, inlier_mask) if m]
    x_out = x_arr[result.macro_outlier_mask]
    y_out = y_arr[result.macro_outlier_mask]

    # Scatter inliers: hue by sensor if multiple, else single color
    plot_df = pd.DataFrame({"x": x_in, "y": y_in, "sensor": sensor_in})
    if len(set(sensor_in)) > 1:
        sns.scatterplot(
            data=plot_df,
            x="x",
            y="y",
            hue="sensor",
            alpha=0.6,
            s=50,
            ax=ax,
            legend="brief",
        )
    else:
        ax.scatter(
            x_in,
            y_in,
            alpha=0.6,
            s=50,
            color="steelblue",
            edgecolors="white",
        )

    # Macro outliers: red X markers
    if len(x_out) > 0:
        ax.scatter(
            x_out,
            y_out,
            marker="X",
            s=80,
            color="crimson",
            edgecolors="darkred",
            linewidths=1.5,
            label=f"Macro outliers (n={result.n_macro_outliers})",
            zorder=6,
        )

    # Raw line (Pass 1, dashed gray)
    x_min, x_max = x_arr.min(), x_arr.max()
    x_line = np.linspace(x_min, x_max, 50)
    y_raw = result.raw_intercept + result.raw_slope * x_line
    ax.plot(
        x_line,
        y_raw,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=(f"Raw (R²={result.raw_batch_r2:.3f}, RMSE={result.raw_batch_rmse:.4f})"),
        zorder=4,
    )

    # Clean line (Pass 2, solid red)
    y_clean = result.intercept + result.slope * x_line
    ax.plot(
        x_line,
        y_clean,
        color="crimson",
        linestyle="-",
        linewidth=2.5,
        label=(
            f"Clean (R²={result.clean_batch_r2:.3f}, RMSE={result.clean_batch_rmse:.4f}, "
            f"n={result.n_points})"
        ),
        zorder=5,
    )

    ax.set_xlabel("Log₁₀ Concentration (CFU/ml)")
    ax.set_ylabel(feature_col.replace("_", " ").title())
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    else:
        ax.set_title(
            f"Macro Batch Regression: {feature_col.replace('_', ' ').title()} "
            f"vs Log Concentration ({serotype}) — Pass sensors only",
            fontweight="bold",
            pad=12,
        )
    ax.legend(loc="best", fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, result
