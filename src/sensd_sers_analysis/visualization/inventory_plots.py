"""Count-preserving visual summaries for the pre-QA data inventory."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns

from sensd_sers_analysis.config.inventory import (
    CHART_COLORS,
    CHART_FONT_SIZE,
    CHART_HEIGHT,
    CHART_WIDTH,
    HEATMAP_CMAP,
    HEATMAP_MIN_HEIGHT,
    HEATMAP_PANEL_WIDTH,
    HEATMAP_ROW_HEIGHT,
)
from sensd_sers_analysis.utils.natural_sort import natural_sort


def plot_inventory_composition(composition: pd.DataFrame) -> Figure:
    """
    Plot spectra by recorded operator with labeled serotype segments.

    Parameters
    ----------
    composition : pandas.DataFrame
        Counts with operator, serotype and spectra columns.

    Returns
    -------
    matplotlib.figure.Figure
        Horizontal stacked bars with zero-based count axis and exact counts.
    """
    table = composition.pivot(index="operator", columns="serotype", values="spectra").fillna(0)
    table = table.loc[table.sum(axis=1).sort_values(ascending=True).index]
    with plt.rc_context({"font.size": CHART_FONT_SIZE}):
        figure, axis = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT), layout="constrained")
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
            ylabel="Recorded operator",
        )
        figure.suptitle("Measured spectra by recorded operator and serotype")
        axis.set_xlim(left=0, right=max(left.max() * 1.12, 1))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend(
            title="Serotype",
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(table.columns),
            frameon=False,
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.18)
    return figure


def plot_inventory_coverage(coverage: pd.DataFrame) -> Figure:
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
    serotypes = natural_sort(coverage["serotype"].unique().tolist())
    sensors = natural_sort(coverage["sensor_id"].unique().tolist())
    concentrations = natural_sort(coverage["target_concentration"].unique().tolist())
    height = max(HEATMAP_MIN_HEIGHT, len(sensors) * HEATMAP_ROW_HEIGHT)
    with plt.rc_context({"font.size": CHART_FONT_SIZE}):
        figure, axes = plt.subplots(
            1,
            len(serotypes),
            figsize=(HEATMAP_PANEL_WIDTH * len(serotypes), height),
            squeeze=False,
            layout="constrained",
        )
        for axis, serotype in zip(axes.flat, serotypes):
            subset = coverage.loc[coverage["serotype"] == serotype]
            matrix = subset.pivot(
                index="sensor_id", columns="target_concentration", values="spectra"
            )
            matrix = matrix.reindex(index=sensors, columns=concentrations).fillna(0)
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
                cbar=axis is axes.flat[-1],
                cbar_kws={"label": "Measured spectra (count)"},
            )
            axis.set(
                title=str(serotype),
                xlabel="Initial target concentration (CFU/mL)",
                ylabel="Sensor ID",
            )
            axis.tick_params(axis="x", rotation=0)
            axis.tick_params(axis="y", rotation=0)
        figure.suptitle("Coverage per sensor and nominal concentration", fontweight="bold")
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
