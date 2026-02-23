"""
SERS spectral plotting.

plot_spectra: line plots of Raman spectra with hue/style grouping, variance bands,
and optional colorbar for continuous hue.
"""

from typing import Optional, Union

import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sensd_sers_analysis.constants import CONCENTRATION_GROUP_ORDER

RAMAN_SHIFT_COL = "raman_shift"
INTENSITY_COL = "intensity"
FILENAME_COL = "filename"
SIGNAL_INDEX_COL = "signal_index"
DEFAULT_NUMERIC_CMAP = "viridis"


def plot_spectra(
    df: pd.DataFrame,
    *,
    hue: Optional[str] = None,
    style: Optional[str] = None,
    show_variance: bool = False,
    errorbar: Union[str, tuple[str, float]] = "sd",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    palette: Optional[Union[str, dict, list]] = None,
    title: Optional[str] = None,
    footnote: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot Raman spectra with hue/style grouping and optional variance bands.

    Args:
        df: Tidy DataFrame with raman_shift, intensity, filename, signal_index.
        hue: Column name for color grouping.
        style: Column name for line style grouping.
        show_variance: If True, mean + error band; if False, individual lines.
        errorbar: "sd", "se", or ("ci", 95). Default: "sd".
        vmin: Lower bound for numeric hue color scale.
        vmax: Upper bound for numeric hue color scale.
        palette: Colormap or color list for categorical hue (e.g., "Set1").
        title: Optional plot title. Auto-generated from hue/style if None.
        footnote: Optional text for subtitle. When show_variance is True,
            variance metadata is appended automatically.
        figsize: Optional (width, height) in inches.
        ax: Optional axes to draw on.

    Returns:
        matplotlib.figure.Figure.
    """
    _validate_data(df)

    spectrum_id = df[FILENAME_COL].astype(str) + "_" + df[SIGNAL_INDEX_COL].astype(str)

    hue_is_numeric = (
        hue is not None and hue in df.columns and df[hue].dtype.kind in ("i", "f")
    )

    if hue_is_numeric:
        norm = _prepare_continuous_hue(df, hue, vmin, vmax)
        plot_palette = DEFAULT_NUMERIC_CMAP
        use_legend = style is not None
    else:
        norm = None
        plot_palette = palette
        use_legend = True

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    plot_kwargs: dict = {
        "data": df,
        "x": RAMAN_SHIFT_COL,
        "y": INTENSITY_COL,
        "ax": ax,
        "legend": use_legend,
    }
    if hue is not None:
        plot_kwargs["hue"] = hue
        if hue == "concentration_group" and hue in df.columns:
            order = [
                v for v in CONCENTRATION_GROUP_ORDER if v in df[hue].astype(str).values
            ]
            if order:
                plot_kwargs["hue_order"] = order
    if style is not None:
        plot_kwargs["style"] = style
        if style == "concentration_group" and style in df.columns:
            order = [
                v
                for v in CONCENTRATION_GROUP_ORDER
                if v in df[style].astype(str).values
            ]
            if order:
                plot_kwargs["style_order"] = order
    if plot_palette is not None:
        plot_kwargs["palette"] = plot_palette
    if norm is not None:
        plot_kwargs["hue_norm"] = norm

    if show_variance:
        plot_kwargs["errorbar"] = errorbar
    else:
        plot_kwargs["units"] = spectrum_id
        plot_kwargs["estimator"] = None

    sns.lineplot(**plot_kwargs)

    if hue_is_numeric and style is not None:
        _filter_legend_to_style_only(ax, df, style)

    resolved_title = title
    if resolved_title is None:
        base = "SERS Spectra"
        if hue and style:
            resolved_title = (
                f"{base} by {_hue_to_label(hue)} and {_hue_to_label(style)}"
            )
        elif hue:
            resolved_title = f"{base} by {_hue_to_label(hue)}"
        elif style:
            resolved_title = f"{base} by {_hue_to_label(style)}"
        else:
            resolved_title = f"{base} Overview"

    subtitle_text: Optional[str] = None
    if show_variance:
        variance_text = _format_errorbar_text(errorbar)
        subtitle_text = f"{footnote}\n{variance_text}" if footnote else variance_text
    elif footnote:
        subtitle_text = footnote
    _apply_aesthetics(ax, title=resolved_title, subtitle_text=subtitle_text)

    if hue_is_numeric and norm is not None:
        _add_colorbar(fig, ax, norm, _hue_to_label(hue))

    return fig


def _prepare_continuous_hue(
    df: pd.DataFrame,
    hue: str,
    vmin_val: Optional[float],
    vmax_val: Optional[float],
) -> matplotlib.colors.Normalize:
    """Return Normalize for numeric hue with fixed scale."""
    vals = df[hue].dropna()
    if len(vals) == 0:
        vmin_val = vmin_val if vmin_val is not None else 0.0
        vmax_val = vmax_val if vmax_val is not None else 1.0
    else:
        vmin_val = vmin_val if vmin_val is not None else float(vals.min())
        vmax_val = vmax_val if vmax_val is not None else float(vals.max())
    return matplotlib.colors.Normalize(vmin=vmin_val, vmax=vmax_val)


def _format_errorbar_text(errorbar: Union[str, tuple]) -> str:
    """Return human-readable description of errorbar for footnote."""
    if errorbar == "sd":
        return "Shaded region: ±1 Standard Deviation"
    if errorbar == "se":
        return "Shaded region: ±1 Standard Error"
    if isinstance(errorbar, tuple) and len(errorbar) >= 2:
        kind, val = errorbar[0], errorbar[1]
        if kind == "ci":
            return f"Shaded region: {val:.0f}% Confidence Interval"
    return f"Shaded region: {errorbar}"


def _hue_to_label(hue_name: str) -> str:
    """Convert hue column name to colorbar label."""
    return hue_name.replace("_", " ").strip().capitalize()


def _validate_data(df: pd.DataFrame) -> None:
    """Ensure DataFrame has required columns for plotting."""
    required = [RAMAN_SHIFT_COL, INTENSITY_COL, FILENAME_COL, SIGNAL_INDEX_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame must contain columns {required}. Missing: {missing}"
        )
    if df.empty:
        raise ValueError("DataFrame is empty")


def _apply_aesthetics(
    ax: plt.Axes,
    *,
    title: Optional[str] = None,
    subtitle_text: Optional[str] = None,
) -> None:
    """Apply standardized labels, legend placement, and clean scientific style."""
    if title is not None:
        ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    if subtitle_text:
        ax.text(
            0.5,
            1.02,
            subtitle_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="gray",
            clip_on=False,
        )
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_loc("best")
    sns.despine(ax=ax)
    ax.grid(True, alpha=0.3, linestyle="--")


def _filter_legend_to_style_only(ax: plt.Axes, df: pd.DataFrame, style: str) -> None:
    """Remove hue entries from legend when hue uses colorbar; keep only style entries."""
    leg = ax.get_legend()
    if leg is None:
        return
    style_values = set(str(v) for v in df[style].dropna().unique())
    handles, labels = ax.get_legend_handles_labels()
    filtered = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label in style_values
    ]
    if not filtered:
        return
    seen: set[str] = set()
    deduped = []
    for handle, label in filtered:
        if label not in seen:
            seen.add(label)
            deduped.append((handle, label))
    if deduped:
        new_handles, new_labels = zip(*deduped)
        leg.remove()
        ax.legend(new_handles, new_labels, loc="best")


def _add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    norm: matplotlib.colors.Normalize,
    label: str,
) -> None:
    """Add colorbar to figure."""
    cmap = matplotlib.colormaps[DEFAULT_NUMERIC_CMAP]
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02, fraction=0.046)
    cbar.set_label(label)
