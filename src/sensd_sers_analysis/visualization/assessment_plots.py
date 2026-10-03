"""
Visualization for sensor assessment: degradation trends and batch comparisons.

Provides publication-ready plots for the Sensor Assessment & Report module.
"""

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sensd_sers_analysis.utils.availability import PlotUnavailableError
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sensd_sers_analysis.assessment.sensor_assessment_regression import (
    ConcentrationRegressionResult,
    MacroRegressionResult,
    compute_macro_batch_regression,
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
        raise ValueError(f"Required columns '{feature_col}' or '{sequence_col}' not in DataFrame.")

    df_clean = df.dropna(subset=[feature_col, sequence_col])
    if df_clean.empty:
        raise PlotUnavailableError("No valid data for degradation plot.")

    if ax is None:
        fig = Figure(figsize=figsize)
        ax = fig.subplots()
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
                if len(x) >= 2 and np.unique(x).size >= 2:
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

        if len(x) >= 2 and np.unique(x).size >= 2:
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
        raise ValueError(f"Required columns '{feature_col}' or '{sensor_col}' not in DataFrame.")

    df_clean = df.dropna(subset=[feature_col])
    if df_clean.empty:
        raise PlotUnavailableError("No valid data for batch boxplot.")

    if ax is None:
        fig = Figure(figsize=figsize)
        ax = fig.subplots()
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
    ax.tick_params(axis="x", rotation=45 if len(df_clean[sensor_col].unique()) > 5 else 0)
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


def plot_sensor_batch_stability(
    df: pd.DataFrame,
    batch_table: pd.DataFrame,
    feature_col: str,
    *,
    sensor_col: str = "sensor_id",
    z_threshold: float = 2.0,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Between-sensor stability: each sensor's mean vs the batch consensus band.

    A boxplot fabricates quartiles/IQR from 1–2 replicates and visually rewards
    single-measurement sensors (a flat line looks "perfect"), while really
    comparing within-sensor spread. This plot instead answers *do the sensors
    agree with each other?*:

    - Individual reads are drawn as jittered dots, so tiny replicate counts are
      shown honestly (one dot means one measurement).
    - Each sensor's mean is a marker, with a ±SD error bar only when it has ≥2
      reads (single-read sensors get no error bar, not a zero-width one).
    - The batch consensus is a horizontal line at ``batch_mean`` with a shaded
      ``batch_mean ± z_threshold·batch_std`` band.
    - Sensors whose mean falls outside the band (``|z_from_batch| > z_threshold``)
      are highlighted as deviating.

    Parameters
    ----------
    df:
        Feature dataframe restricted to the analyzed serotype/target.
    batch_table:
        Output of :func:`compute_batch_variance` (one row per sensor with
        ``mean``, ``std``, ``n_samples``, ``batch_mean``, ``batch_std``,
        ``z_from_batch``).
    feature_col:
        Feature plotted on the y-axis.
    sensor_col:
        Sensor identifier column.
    z_threshold:
        Deviation cutoff for highlighting and the band width.
    title, figsize, ax:
        Standard matplotlib overrides.

    Returns
    -------
    plt.Figure
        The rendered figure.
    """
    required = [feature_col, sensor_col]
    if any(c not in df.columns for c in required):
        raise ValueError(f"Required columns '{feature_col}' or '{sensor_col}' not in DataFrame.")
    if batch_table.empty:
        raise PlotUnavailableError("No batch variance rows to plot.")

    df_clean = df.dropna(subset=[feature_col])
    if df_clean.empty:
        raise PlotUnavailableError("No valid data for sensor batch stability plot.")

    if ax is None:
        fig = Figure(figsize=figsize)
        ax = fig.subplots()
    else:
        fig = ax.get_figure()

    ordered = batch_table.sort_values("mean", na_position="last").reset_index(drop=True)
    batch_mean = float(ordered["batch_mean"].iloc[0]) if "batch_mean" in ordered.columns else np.nan
    batch_std = float(ordered["batch_std"].iloc[0]) if "batch_std" in ordered.columns else np.nan

    # Batch consensus line + deviation band.
    if np.isfinite(batch_mean):
        ax.axhline(
            batch_mean,
            color="black",
            linestyle="-",
            linewidth=1.5,
            alpha=0.7,
            label="Batch mean",
            zorder=2,
        )
    if np.isfinite(batch_mean) and np.isfinite(batch_std) and batch_std > 0:
        ax.axhspan(
            batch_mean - z_threshold * batch_std,
            batch_mean + z_threshold * batch_std,
            color="green",
            alpha=0.08,
            label=f"±{z_threshold:g}·batch SD",
            zorder=1,
        )

    rng = np.random.default_rng(0)
    deviating_labelled = False
    for position, (_, row) in enumerate(ordered.iterrows()):
        sensor = row[sensor_col]
        sensor_mean = float(row["mean"]) if np.isfinite(row["mean"]) else np.nan
        sensor_std = float(row["std"]) if "std" in row and np.isfinite(row["std"]) else np.nan
        n_samples = int(row["n_samples"]) if "n_samples" in row else 0
        z_val = float(row["z_from_batch"]) if "z_from_batch" in row else np.nan
        is_deviating = np.isfinite(z_val) and abs(z_val) > z_threshold
        color = "crimson" if is_deviating else "steelblue"

        points = df_clean.loc[df_clean[sensor_col] == sensor, feature_col].astype(float).values
        if len(points) > 0:
            jitter = rng.uniform(-0.12, 0.12, size=len(points))
            ax.scatter(
                np.full(len(points), position) + jitter,
                points,
                s=28,
                color=color,
                alpha=0.35,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )

        # Mean marker with a ±SD error bar only when spread is defined (n >= 2).
        yerr = sensor_std if (n_samples >= 2 and np.isfinite(sensor_std)) else None
        ax.errorbar(
            position,
            sensor_mean,
            yerr=yerr,
            fmt="D",
            markersize=8,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            zorder=4,
            label=("Deviating sensor" if (is_deviating and not deviating_labelled) else None),
        )
        if is_deviating:
            deviating_labelled = True
        if n_samples < 2:
            ax.annotate(
                "n=1",
                (position, sensor_mean),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=7,
                color="gray",
                va="center",
            )

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered[sensor_col].astype(str).tolist())
    ax.tick_params(axis="x", rotation=45 if len(ordered) > 5 else 0)
    ax.set_xlabel(sensor_col.replace("_", " ").title())
    ax.set_ylabel(feature_col.replace("_", " ").title())
    ax.set_title(
        title
        or (
            f"Between-Sensor Stability: {feature_col.replace('_', ' ').title()} "
            "(mean ± SD vs batch band)"
        ),
        fontweight="bold",
        pad=12,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=8, framealpha=0.85)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_signal_vs_concentration_cv(
    consistency_table: pd.DataFrame,
    *,
    sensor_col: str = "sensor_id",
    feature_col: str = "feature",
    signal_cv_col: str = "cv_filtered",
    conc_cv_col: str = "conc_cv_raw",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (11, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Grouped bars of signal CV% (per feature) vs sample concentration CV% per sensor.

    This makes the sensor-added variability explicit: for each sensor, one bar
    per feature shows the SERS **signal** CV%, and a final distinct bar shows the
    actual-CFU **concentration** CV% (the inherent sample spread, shared across
    features). The two CVs compare observed relative spread; their difference does not
    identify a sensor variance component or establish its cause.

    Undefined CVs (e.g. a single replicate) are NaN and simply draw no bar,
    rather than a misleading zero-height bar.

    Parameters
    ----------
    consistency_table:
        Long per-(sensor, feature) table from
        :func:`get_consistency_summary_table` with ``signal_cv_col`` and
        ``conc_cv_col`` as fractions.
    sensor_col, feature_col:
        Grouping columns.
    signal_cv_col:
        Signal CV column (fraction) plotted per feature.
    conc_cv_col:
        Concentration CV column (fraction), feature-independent per sensor.
    title, figsize, ax:
        Standard matplotlib overrides.

    Returns
    -------
    plt.Figure
        The rendered figure.
    """
    required = [sensor_col, feature_col, signal_cv_col]
    if any(c not in consistency_table.columns for c in required):
        raise ValueError(f"consistency_table missing one of {required}.")
    if consistency_table.empty:
        raise PlotUnavailableError("No consistency rows to plot.")

    sensors = consistency_table[sensor_col].astype(str).drop_duplicates().tolist()
    features = consistency_table[feature_col].astype(str).drop_duplicates().tolist()
    has_conc = conc_cv_col in consistency_table.columns

    # Signal CV% per (sensor, feature); concentration CV% per sensor (first row).
    signal_pct: dict[tuple[str, str], float] = {}
    conc_pct: dict[str, float] = {}
    for _, row in consistency_table.iterrows():
        sensor = str(row[sensor_col])
        feature = str(row[feature_col])
        signal_pct[(sensor, feature)] = float(row[signal_cv_col]) * 100
        if has_conc and sensor not in conc_pct:
            conc_pct[sensor] = float(row[conc_cv_col]) * 100

    if ax is None:
        fig = Figure(figsize=figsize)
        ax = fig.subplots()
    else:
        fig = ax.get_figure()

    n_groups = len(features) + (1 if has_conc else 0)
    bar_width = 0.8 / max(n_groups, 1)
    x = np.arange(len(sensors), dtype=float)
    palette = sns.color_palette("muted", n_colors=max(len(features), 1))

    for i, feature in enumerate(features):
        heights = [signal_pct.get((s, feature), np.nan) for s in sensors]
        ax.bar(
            x + (i - (n_groups - 1) / 2) * bar_width,
            heights,
            bar_width,
            color=palette[i],
            label=f"Signal: {feature}",
            zorder=3,
        )

    if has_conc:
        heights = [conc_pct.get(s, np.nan) for s in sensors]
        ax.bar(
            x + (len(features) - (n_groups - 1) / 2) * bar_width,
            heights,
            bar_width,
            color="dimgray",
            hatch="//",
            edgecolor="white",
            label="Concentration (sample spread)",
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sensors)
    ax.tick_params(axis="x", rotation=45 if len(sensors) > 5 else 0)
    ax.set_xlabel(sensor_col.replace("_", " ").title())
    ax.set_ylabel("CV (%)")
    ax.set_title(
        title or "Signal CV% vs Sample Concentration CV% by Sensor",
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
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
        raise ValueError(f"Required columns '{feature_col}' or '{log_conc_col}' not in DataFrame.")

    valid = df[[log_conc_col, feature_col]].notna().all(axis=1)
    df_plot = df.loc[valid]

    if df_plot.empty:
        raise PlotUnavailableError("No valid data for concentration regression plot.")

    if ax is None:
        fig = Figure(figsize=figsize)
        ax = fig.subplots()
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
        ax.scatter(x, y, alpha=0.6, s=50, color="steelblue", edgecolors="white", zorder=3)

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
            f"Sensor assessment: {feature_col.replace('_', ' ').title()} vs log concentration",
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
        raise PlotUnavailableError(
            f"No valid data for serotype={serotype}. Need rows with non-null "
            f"{log_conc_col} and {feature_col}."
        )

    sensors = subset[sensor_col].dropna().unique()
    if len(sensors) == 0:
        raise PlotUnavailableError(f"No sensor_id values found for serotype={serotype}.")

    excluded = excluded_sensors or set()
    excluded = {str(s) for s in excluded}

    if ax is None:
        fig = Figure(figsize=figsize)
        ax = fig.subplots()
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

        if len(x) >= 2 and np.unique(x).size >= 2:
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
    """Plot pooled sensor responses and their raw and cleaned macro fits.

    Parameters
    ----------
    df : pd.DataFrame
        Feature data with identity, predictor, and response columns.
    serotype : str
        Serotype to include when computing a result.
    feature_col : str
        Response column and axis label.
    pass_sensors : set[str]
        Sensor IDs selected by the caller's QA policy.
    sensor_col, serotype_col, log_conc_col : str
        Identity and predictor column names.
    macro_result : MacroRegressionResult, optional
        Fit artifact for the selected data. Its pooled points, labels, masks,
        and coefficients are used directly. If absent, compute the artifact once.
    title : str, optional
        Plot title.
    figsize : tuple[float, float]
        Figure size in inches when creating axes.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    tuple[matplotlib.figure.Figure, MacroRegressionResult or None]
        Figure and fitted artifact. An unavailable fit produces an explanatory plot.

    Raises
    ------
    ValueError
        If required columns or valid responses for the requested serotype are absent.
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
        raise PlotUnavailableError(
            f"No valid data for serotype={serotype}. Need non-null "
            f"{log_conc_col} and {feature_col}."
        )

    result = (
        macro_result
        if macro_result is not None
        else compute_macro_batch_regression(
            df,
            serotype,
            feature_col,
            pass_sensors,
            sensor_col=sensor_col,
            serotype_col=serotype_col,
            log_conc_col=log_conc_col,
        )
    )

    if result is None:
        if ax is None:
            fig = Figure(figsize=figsize)
            ax = fig.subplots()
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
        fig = Figure(figsize=figsize)
        ax = fig.subplots()
    else:
        fig = ax.get_figure()

    # Coordinates, sensor labels, and masks come from the same fitted artifact.
    x_arr = result.x_pooled
    y_arr = result.y_pooled
    inlier_mask = ~result.macro_outlier_mask
    x_in = x_arr[inlier_mask]
    y_in = y_arr[inlier_mask]
    sensor_in = result.pooled_sensor_ids[inlier_mask]
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
