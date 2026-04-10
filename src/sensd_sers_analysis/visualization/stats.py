"""
Statistical plotting for extracted scalar features.

Handles discrete feature distributions (boxplots, violin plots) for
comparing metrics across groups (e.g., serotype, concentration).
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sensd_sers_analysis.utils.natural_sort import order_concentration_labels


def plot_feature_distribution(
    df_features: pd.DataFrame,
    feature_col: str,
    *,
    x: Optional[str] = None,
    hue: Optional[str] = None,
    plot_type: str = "box",
    show_points: bool = True,
    max_points_for_strip: int = 500,
    title: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot distribution of an extracted scalar feature by grouping variable(s).

    Uses boxplot or violin plot with optional stripplot overlay for
    individual data points (when sample size is manageable).

    Boxplot outliers (fliers): Points outside Q1 - 1.5*IQR or Q3 + 1.5*IQR,
    where IQR = Q3 - Q1. Controlled by seaborn's ``whis`` parameter (default 1.5).

    Args:
        df_features: DataFrame from extract_basic_features (metadata +
            max_intensity, mean_intensity, integral_area).
        feature_col: Name of the feature column to plot (y-axis).
        x: Column for x-axis grouping (e.g., "concentration_group",
            "concentration"). If None, uses a single anonymous group.
        hue: Optional column for color grouping (e.g., "serotype").
        plot_type: "box" for boxplot, "violin" for violin plot.
        show_points: If True, overlay stripplot when n_samples <=
            max_points_for_strip.
        max_points_for_strip: Maximum sample size for stripplot overlay.
        title: Optional plot title. Auto-generated from feature/x/hue if None.
        figsize: Optional (width, height) in inches.
        ax: Optional axes to draw on.

    Returns:
        matplotlib.figure.Figure.

    Example:
        >>> df_feat = extract_basic_features(load_sers_data("data/"))
        >>> fig = plot_feature_distribution(
        ...     df_feat, "integral_area",
        ...     x="concentration_group", hue="serotype",
        ... )
    """
    if feature_col not in df_features.columns:
        raise ValueError(
            f"feature_col '{feature_col}' not in DataFrame. Available: {list(df_features.columns)}"
        )
    if df_features.empty:
        raise ValueError("DataFrame is empty")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Drop rows with NaN in feature to avoid seaborn boxprops UnboundLocalError
    # (seaborn bug when groups have no valid data)
    df_clean = df_features.dropna(subset=[feature_col])
    if df_clean.empty:
        # Try fallback features when the selected one has no valid data
        fallback_cols = [
            c
            for c in ["max_intensity", "mean_intensity", "integral_area"]
            if c in df_features.columns and c != feature_col and df_features[c].notna().any()
        ]
        if fallback_cols:
            raise ValueError(
                f"All values in '{feature_col}' are NaN. Try plotting '{fallback_cols[0]}' instead."
            )
        raise ValueError(
            "All feature values are NaN. Check that the loaded data contains "
            "Raman intensity columns (rs_*) with valid numeric values."
        )

    x_col = x if x is not None else "_single"
    if x_col == "_single":
        df = df_clean.copy()
        df[x_col] = "all"
    else:
        if x not in df_features.columns:
            raise ValueError(f"x column '{x}' not in DataFrame")
        df = df_clean.copy()

    def _conc_order(col: str) -> list | None:
        if col != "concentration_group" or col not in df.columns:
            return None
        vals = df[col].astype(str).dropna().unique().tolist()
        vals = [v for v in vals if v]
        return order_concentration_labels(vals) if vals else None

    plot_kwargs: dict = {
        "data": df,
        "x": x_col,
        "y": feature_col,
        "ax": ax,
        "legend": hue is not None,
    }
    if x_col != "_single":
        x_order = _conc_order(x_col)
        if x_order is not None:
            plot_kwargs["order"] = x_order
    if hue is not None:
        if hue not in df.columns:
            raise ValueError(f"hue column '{hue}' not in DataFrame")
        plot_kwargs["hue"] = hue
        hue_order = _conc_order(hue)
        if hue_order is not None:
            plot_kwargs["hue_order"] = hue_order

    if plot_type == "violin":
        sns.violinplot(**plot_kwargs)
    else:
        try:
            sns.boxplot(**plot_kwargs)
        except UnboundLocalError as e:
            if "boxprops" in str(e):
                # Seaborn bug when certain group combinations yield no plottable boxes;
                # fall back to violin plot which does not have this issue
                sns.violinplot(**plot_kwargs)
            else:
                raise

    if show_points and len(df) <= max_points_for_strip:
        strip_kw: dict = {
            "data": df,
            "x": x_col,
            "y": feature_col,
            "ax": ax,
            "alpha": 0.35,
            "size": 3,
            "jitter": 0,
            "dodge": hue is not None,
            "legend": False,
        }
        if x_col != "_single" and (x_order := plot_kwargs.get("order")):
            strip_kw["order"] = x_order
        if hue is not None:
            strip_kw["hue"] = hue
            if hue_order := plot_kwargs.get("hue_order"):
                strip_kw["hue_order"] = hue_order
        else:
            strip_kw["hue"] = x_col
            strip_kw["palette"] = {v: "black" for v in df[x_col].unique()}
        sns.stripplot(**strip_kw)

    resolved_title = title
    if resolved_title is None:
        parts = [feature_col.replace("_", " ").title()]
        if x and x != "_single":
            parts.append(f"by {x.replace('_', ' ').title()}")
        if hue:
            parts.append(f"(hue: {hue})")
        resolved_title = " — ".join(parts)
    ax.set_title(resolved_title, pad=20, fontsize=12, fontweight="bold")
    ax.set_ylabel(feature_col.replace("_", " ").title())
    if x_col == "_single":
        ax.set_xlabel("")
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(x.replace("_", " ").title())

    leg = ax.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1.02, 1))
        leg.set_loc("upper left")

    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_feature_correlation_heatmap(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    *,
    method: str = "pearson",
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot a symmetric correlation matrix for selected numeric feature columns.

    Uses pairwise-complete observations. Columns with fewer than two finite
    values after dropping NaNs are omitted from the matrix.

    Parameters
    ----------
    df_features:
        Sample-level feature table.
    feature_cols:
        Column names to include (must exist in ``df_features``).
    method:
        Correlation method passed to :meth:`pandas.DataFrame.corr`.
    figsize:
        Figure size in inches when ``ax`` is not provided.
    ax:
        Optional axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the heatmap.

    Raises
    ------
    ValueError
        If fewer than two valid columns remain for correlation.
    """
    cols = [c for c in feature_cols if c in df_features.columns]
    if len(cols) < 2:
        raise ValueError("Need at least two feature columns present in the DataFrame.")

    sub = df_features[cols].apply(pd.to_numeric, errors="coerce")
    valid_cols = [c for c in cols if sub[c].notna().sum() >= 2]
    if len(valid_cols) < 2:
        raise ValueError("Need at least two feature columns with ≥2 finite values for correlation.")
    corr = sub[valid_cols].corr(method=method, min_periods=2)
    if corr.empty:
        raise ValueError("Correlation matrix is empty.")

    if ax is None:
        if figsize is not None:
            w, h = figsize
        else:
            w = max(6.0, 0.65 * len(valid_cols))
            h = max(5.0, 0.65 * len(valid_cols))
        fig, ax = plt.subplots(figsize=(w, h))
    else:
        fig = ax.get_figure()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": f"{method.title()} r"},
        ax=ax,
    )
    ax.set_title(
        f"Feature correlation ({method}) — lower triangle",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    return fig


def compute_feature_log_concentration_correlations(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    *,
    log_conc_col: str = "log_concentration",
) -> pd.DataFrame:
    """
    Pearson correlation and sample size per feature vs log concentration.

    Rows with fewer than three paired finite observations or degenerate
    variance in either axis are omitted.

    Parameters
    ----------
    df_features:
        Sample-level feature table with ``log_conc_col``.
    feature_cols:
        Feature columns to evaluate.
    log_conc_col:
        Numeric concentration axis (typically log₁₀ CFU/ml).

    Returns
    -------
    pandas.DataFrame
        Columns: ``feature``, ``pearson_r``, ``p_value``, ``n`` sorted by ``|pearson_r|`` descending.
    """
    if log_conc_col not in df_features.columns:
        raise ValueError(f"Column '{log_conc_col}' not in DataFrame.")

    records: list[tuple[str, float, float, int]] = []
    for col in feature_cols:
        if col not in df_features.columns:
            continue
        paired = df_features[[log_conc_col, col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(paired) < 3:
            continue
        x = paired[log_conc_col].astype(float).values
        y = paired[col].astype(float).values
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        r, p = stats.pearsonr(x, y)
        records.append((col, float(r), float(p), len(paired)))

    if not records:
        return pd.DataFrame(columns=["feature", "pearson_r", "p_value", "n"])

    records.sort(key=lambda t: abs(t[1]), reverse=True)
    return pd.DataFrame(
        records,
        columns=["feature", "pearson_r", "p_value", "n"],
    )


def plot_feature_log_concentration_correlation_bars(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    *,
    log_conc_col: str = "log_concentration",
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Bar chart of Pearson correlation between each feature and log concentration.

    For each feature, correlation uses all rows with finite values in both
    ``log_conc_col`` and the feature. Features with fewer than three paired
    observations are skipped.

    Parameters
    ----------
    df_features:
        Sample-level feature table with ``log_conc_col``.
    feature_cols:
        Feature columns to evaluate.
    log_conc_col:
        Column for log₁₀ concentration (or compatible numeric concentration axis).

    Returns
    -------
    matplotlib.figure.Figure
        Horizontal bar chart sorted by correlation magnitude.

    Raises
    ------
    ValueError
        If no feature yields a valid correlation.
    """
    corr_df = compute_feature_log_concentration_correlations(
        df_features,
        feature_cols,
        log_conc_col=log_conc_col,
    )
    if corr_df.empty:
        raise ValueError(
            "No correlations computed; need ≥3 paired finite values per feature "
            f"with non-degenerate '{log_conc_col}' and feature."
        )

    labels = corr_df["feature"].tolist()
    rs = corr_df["pearson_r"].astype(float).tolist()

    if ax is None:
        height = figsize[1] if figsize is not None else max(4.0, 0.35 * len(labels))
        width = figsize[0] if figsize is not None else 10.0
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.get_figure()

    colors = ["steelblue" if r >= 0 else "coral" for r in rs]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, rs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos, labels)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="-", alpha=0.6)
    ax.set_xlabel(f"Pearson r vs {log_conc_col}")
    ax.set_title(
        "Feature — log concentration association (Pearson r)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.invert_yaxis()
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_feature_log_concentration_scatter(
    df_features: pd.DataFrame,
    feature_col: str,
    *,
    log_conc_col: str = "log_concentration",
    hue_col: Optional[str] = None,
    figsize: tuple[float, float] = (10.0, 6.0),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Scatter plot of log concentration vs a single feature with an OLS line.

    Fits :func:`scipy.stats.linregress` on all finite pairs; intended for
    exploratory analysis (not outlier-robust regression).

    Parameters
    ----------
    df_features:
        Sample-level feature table.
    feature_col:
        Feature on the y-axis.
    log_conc_col:
        Log concentration column on the x-axis.
    hue_col:
        Optional categorical column for color grouping (legend).

    Returns
    -------
    matplotlib.figure.Figure
        Scatter with regression line and summary statistics in the title.
    """
    need = [log_conc_col, feature_col]
    for c in need:
        if c not in df_features.columns:
            raise ValueError(f"Required column '{c}' not in DataFrame.")
    if hue_col is not None and hue_col not in df_features.columns:
        raise ValueError(f"hue_col '{hue_col}' not in DataFrame.")

    plot_df = df_features[[log_conc_col, feature_col]].copy()
    plot_df[log_conc_col] = pd.to_numeric(plot_df[log_conc_col], errors="coerce")
    plot_df[feature_col] = pd.to_numeric(plot_df[feature_col], errors="coerce")
    if hue_col:
        plot_df[hue_col] = df_features[hue_col]
    plot_df = plot_df.dropna(subset=[log_conc_col, feature_col])
    if plot_df.empty:
        raise ValueError("No valid rows after dropping NaN in concentration or feature.")

    x = plot_df[log_conc_col].astype(float).values
    y = plot_df[feature_col].astype(float).values
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        raise ValueError("Cannot fit OLS: zero variance in log concentration or feature.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if hue_col and plot_df[hue_col].notna().any():
        for name, grp in plot_df.groupby(hue_col, dropna=True):
            gx = grp[log_conc_col].astype(float).values
            gy = grp[feature_col].astype(float).values
            ax.scatter(
                gx,
                gy,
                alpha=0.65,
                s=45,
                edgecolors="white",
                label=str(name),
                zorder=3,
            )
    else:
        ax.scatter(x, y, alpha=0.65, s=50, color="steelblue", edgecolors="white", zorder=3)

    lr = stats.linregress(x.astype(float), y.astype(float))
    x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 80)
    y_line = lr.intercept + lr.slope * x_line
    ax.plot(
        x_line,
        y_line,
        color="crimson",
        linewidth=2,
        label=f"OLS (R²={lr.rvalue**2:.3f}, n={len(x)})",
        zorder=2,
    )

    ax.set_xlabel("Log₁₀ concentration (axis: " + log_conc_col.replace("_", " ") + ")")
    ax.set_ylabel(feature_col.replace("_", " ").title())
    ax.set_title(
        f"{feature_col.replace('_', ' ').title()} vs {log_conc_col} — exploratory OLS",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="best", fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
