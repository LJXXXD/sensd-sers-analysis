"""Count-preserving visual summaries for the pre-QA data inventory."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns

from sensd_sers_analysis.config.inventory import (
    CHART_COLORS,
    INVENTORY_GROUP_LABELS,
    CHART_FONT_SIZE,
    CHART_HEIGHT,
    CHART_WIDTH,
    HEATMAP_CMAP,
    HEATMAP_MIN_HEIGHT,
    HEATMAP_PANEL_WIDTH,
    HEATMAP_ROW_HEIGHT,
)
from sensd_sers_analysis.utils.natural_sort import natural_sort


def plot_inventory_composition(
    composition: pd.DataFrame, *, group_by: str = "operator", color_by: str | None = "serotype"
) -> Figure:
    """
    Plot spectrum counts grouped by one dimension and colored by another.

    Parameters
    ----------
    composition : pandas.DataFrame
        Counts with group_by, color_by and spectra columns.
    color_by : str or None
        Segment dimension; None displays one color and totals per group.

    Returns
    -------
    matplotlib.figure.Figure
        Horizontal stacked bars with zero-based count axis and exact counts.
    """
    segment = color_by
    if segment == group_by:
        raise ValueError("Group and color dimensions must differ.")
    if segment is None:
        table = (
            composition.groupby(group_by)[["spectra"]].sum().rename(columns={"spectra": "Spectra"})
        )
    else:
        table = composition.pivot(index=group_by, columns=segment, values="spectra").fillna(0)
    table = table.reindex(columns=natural_sort(table.columns.tolist()))
    label = INVENTORY_GROUP_LABELS[group_by]
    if group_by == "month":
        table = table.reindex(index=natural_sort(table.index.tolist())[::-1])
    else:
        table = table.loc[table.sum(axis=1).sort_values(ascending=True).index]
    with plt.rc_context({"font.size": CHART_FONT_SIZE}):
        figure, axis = plt.subplots(
            figsize=(CHART_WIDTH, max(CHART_HEIGHT, len(table) * HEATMAP_ROW_HEIGHT)),
            layout="constrained",
        )
        left = pd.Series(0, index=table.index)
        for index, serotype in enumerate(table.columns):
            values = table[serotype]
            bars = axis.barh(
                table.index,
                values,
                left=left,
                label=str(serotype),
                color=CHART_COLORS[index % len(CHART_COLORS)],
            )
            axis.bar_label(
                bars,
                labels=[str(int(value)) if value else "" for value in values],
                label_type="center",
                color="white",
                fontweight="bold",
            )
            left += values
        for index, value in enumerate(left):
            axis.annotate(
                str(int(value)),
                (value, index),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontweight="bold",
            )
        axis.set(
            xlabel="Measured spectra (count)",
            ylabel=label,
        )
        figure.suptitle(f"Measured spectra by {label.lower()}")
        axis.set_xlim(left=0, right=max(left.max() * 1.12, 1))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend(
            title=INVENTORY_GROUP_LABELS[segment] if segment else None,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(table.columns), 6),
            frameon=False,
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.18)
    return figure


def plot_inventory_coverage(
    coverage: pd.DataFrame,
    *,
    row_dimension: str = "sensor_id",
    column_dimension: str = "target_concentration",
    facet_dimension: str = "serotype",
) -> Figure:
    """
    Plot sensor-by-concentration count matrices with a common scale per serotype.

    Parameters
    ----------
    coverage : pandas.DataFrame
        Observed sensor_id, serotype, target_concentration and spectra counts.

    Returns
    -------
    matplotlib.figure.Figure
        Facets share sensor order, concentration order and color limits.
        Dashes mark no spectrum in the selected data, not failed devices.
    """
    from math import ceil
    from sensd_sers_analysis.config.inventory import COVERAGE_FACET_COLUMNS

    facets = natural_sort(coverage[facet_dimension].unique().tolist())
    rows = natural_sort(coverage[row_dimension].unique().tolist())
    columns = natural_sort(coverage[column_dimension].unique().tolist())
    height = max(HEATMAP_MIN_HEIGHT, len(rows) * HEATMAP_ROW_HEIGHT)
    n_columns = min(len(facets), COVERAGE_FACET_COLUMNS)
    n_rows = ceil(len(facets) / n_columns)
    with plt.rc_context({"font.size": CHART_FONT_SIZE}):
        figure, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(HEATMAP_PANEL_WIDTH * n_columns, height * n_rows),
            squeeze=False,
            layout="constrained",
        )
        for axis, facet in zip(axes.flat, facets):
            subset = coverage.loc[coverage[facet_dimension] == facet]
            matrix = subset.pivot(index=row_dimension, columns=column_dimension, values="spectra")
            matrix = matrix.reindex(index=rows, columns=columns).fillna(0)
            labels = matrix.astype(int).astype(str).mask(matrix.eq(0), "–")
            sns.heatmap(
                matrix,
                ax=axis,
                annot=labels,
                fmt="",
                cmap=HEATMAP_CMAP,
                vmin=0,
                vmax=max(coverage["spectra"].max(), 1),
                linewidths=0.5,
                cbar=False,
            )
            axis.set(
                title=f"{INVENTORY_GROUP_LABELS[facet_dimension]}: {facet}",
                xlabel=INVENTORY_GROUP_LABELS[column_dimension],
                ylabel=INVENTORY_GROUP_LABELS[row_dimension],
            )
            axis.tick_params(axis="x", rotation=45 if len(columns) > 6 else 0)
            axis.tick_params(axis="y", rotation=0)
        for axis in list(axes.flat)[len(facets) :]:
            axis.set_visible(False)
        figure.colorbar(
            axes.flat[0].collections[0], ax=list(axes.flat), label="Number of spectra", shrink=0.5
        )
        figure.suptitle("Measurement coverage · Counts per combination", fontweight="bold")
    return figure


def plot_inventory_timeline(timeline: pd.DataFrame) -> Figure:
    """
    Plot observed spectra per acquisition month, including internal empty months.

    Parameters
    ----------
    timeline : pandas.DataFrame
        Calendar month labels and measured-spectrum counts.

    Returns
    -------
    matplotlib.figure.Figure
        Monthly bars; missing acquisition dates are reported separately by UI.
    """
    with plt.rc_context({"font.size": CHART_FONT_SIZE}):
        figure, axis = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT), layout="constrained")
        bars = axis.bar(
            timeline["month"], timeline["spectra"], color=CHART_COLORS[0], label="Dated spectra"
        )
        axis.bar_label(bars, padding=3)
        axis.set(
            title="Measured spectra by acquisition month",
            xlabel="Acquisition month",
            ylabel="Measured spectra (count)",
        )
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.margins(y=0.15)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.18)
    return figure


def plot_screening_counts(counts: pd.DataFrame) -> Figure:
    """Show sensor-screening outcomes with counts conserved across all statuses."""
    from sensd_sers_analysis.config.inventory import SCREENING_COLORS

    table = counts.pivot(index="sensor_id", columns="status", values="spectra").fillna(0)
    table = table.reindex(index=natural_sort(table.index.astype(str).tolist()))
    with plt.rc_context({"font.size": CHART_FONT_SIZE}):
        figure, axis = plt.subplots(
            figsize=(CHART_WIDTH, max(CHART_HEIGHT, len(table) * HEATMAP_ROW_HEIGHT)),
            layout="constrained",
        )
        left = pd.Series(0.0, index=table.index)
        for status, color in SCREENING_COLORS.items():
            values = table.get(status, pd.Series(0, index=table.index))
            axis.barh(
                table.index,
                values,
                left=left,
                color=color,
                label=status,
                hatch="///" if status == "Excluded" else None,
            )
            left += values
        for index, value in enumerate(left):
            axis.annotate(
                str(int(value)),
                (value, index),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
            )
        axis.invert_yaxis()
        axis.set(
            xlabel="Number of spectra",
            ylabel="Sensor ID",
            title="Sensor screening: spectra passing or excluded",
        )
        axis.set_xlim(0, max(left.max() * 1.12, 1))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    return figure
