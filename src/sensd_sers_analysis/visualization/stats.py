"""
Statistical plotting for extracted scalar features.

Handles discrete feature distributions (boxplots, violin plots) for
comparing metrics across groups (e.g., serotype, concentration).
"""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
            f"feature_col '{feature_col}' not in DataFrame. "
            f"Available: {list(df_features.columns)}"
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
            if c in df_features.columns
            and c != feature_col
            and df_features[c].notna().any()
        ]
        if fallback_cols:
            raise ValueError(
                f"All values in '{feature_col}' are NaN. Try plotting "
                f"'{fallback_cols[0]}' instead."
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
            "color": "black",
            "alpha": 0.35,
            "size": 3,
            "jitter": 0,
            "dodge": hue is not None,
        }
        if x_col != "_single" and (x_order := plot_kwargs.get("order")):
            strip_kw["order"] = x_order
        if hue is not None:
            strip_kw["hue"] = hue
            strip_kw["legend"] = False
            if hue_order := plot_kwargs.get("hue_order"):
                strip_kw["hue_order"] = hue_order
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
