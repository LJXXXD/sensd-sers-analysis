"""
Spectra Viewer tab — plot SERS spectra with hue, style, and variance options.
"""

import logging

import streamlit as st

from components.shared_ui import render_figure_stretch
from sensd_sers_analysis.processing import get_plot_hue_columns, pick_preferred_column
from sensd_sers_analysis.utils import format_column_label
from sensd_sers_analysis.visualization import VARIANCE_OPTIONS, plot_spectra
from theme import (
    DEFAULT_FIGSIZE_WIDTH,
    PLOT_HEIGHT_DEFAULT,
    PLOT_HEIGHT_MAX,
    PLOT_HEIGHT_MIN,
)

logger = logging.getLogger(__name__)


def render(filtered):
    """Render the Spectral Viewer tab."""
    available_cols = get_plot_hue_columns(filtered)
    hue_options = ["None"] + available_cols
    hue_default = pick_preferred_column(available_cols) or "None"

    hue_default = "serotype" if "serotype" in available_cols else hue_default
    variance_labels = [v[0] for v in VARIANCE_OPTIONS]
    color, display = st.columns(2)
    with color:
        hue_choice = st.selectbox(
            "Color by",
            hue_options,
            index=hue_options.index(hue_default),
            format_func=format_column_label,
            key="spectra_color",
        )
    with display:
        variance_choice = st.selectbox("Display", variance_labels, key="spectra_display")
    with st.expander("More plot options"):
        style_choice = st.selectbox(
            "Line style by",
            ["None"] + available_cols,
            format_func=format_column_label,
            key="spectra_style",
        )
        plot_height = st.slider(
            "Height (in)",
            min_value=PLOT_HEIGHT_MIN,
            max_value=PLOT_HEIGHT_MAX,
            value=PLOT_HEIGHT_DEFAULT,
            step=1,
            key="spectra_height",
        )

    _vo = VARIANCE_OPTIONS[variance_labels.index(variance_choice)]
    show_variance = _vo[1]
    errorbar = _vo[2]
    hue_col = None if hue_choice == "None" else hue_choice
    style_col = None if style_choice == "None" else style_choice

    try:
        fig = plot_spectra(
            filtered,
            hue=hue_col,
            style=style_col,
            show_variance=show_variance,
            errorbar=errorbar,
            figsize=(DEFAULT_FIGSIZE_WIDTH, plot_height),
        )
        render_figure_stretch(fig)
    except ValueError as e:
        logger.warning("Spectra plot error: %s", e)
        st.error(f"Plot error: {e}")
        st.caption(
            "Ensure filtered data has required columns: raman_shift, intensity, "
            "filename, signal_index."
        )
