"""
Spectra Viewer tab — plot SERS spectra with hue, style, and variance options.
"""

import streamlit as st

from components.shared_ui import render_figure_stretch
from theme import (
    DEFAULT_FIGSIZE_WIDTH,
    PLOT_HEIGHT_DEFAULT,
    PLOT_HEIGHT_MAX,
    PLOT_HEIGHT_MIN,
)

from sensd_sers_analysis.processing import (
    get_plot_hue_columns,
    pick_preferred_column,
)
from sensd_sers_analysis.visualization import VARIANCE_OPTIONS, plot_spectra


def render(filtered):
    """Render the Spectral Viewer tab."""
    available_cols = get_plot_hue_columns(filtered)
    hue_options = ["None"] + available_cols
    hue_default = pick_preferred_column(available_cols) or "None"

    st.markdown("#### Spectra plot options")
    c_hue, c_style, c_var, c_h = st.columns(4)
    with c_hue:
        hue_choice = st.radio(
            "Hue _(color)_",
            options=hue_options,
            index=hue_options.index(hue_default),
            horizontal=False,
            key="hue_radio",
        )
    with c_style:
        style_choice = st.radio(
            "Style _(line)_",
            options=["None"] + available_cols,
            index=0,
            horizontal=False,
            key="style_radio",
        )
    with c_var:
        variance_labels = [v[0] for v in VARIANCE_OPTIONS]
        variance_choice = st.radio(
            "Display",
            options=variance_labels,
            index=0,
            horizontal=False,
            key="variance_radio",
        )
    with c_h:
        plot_height = st.slider(
            "Height (in)",
            min_value=PLOT_HEIGHT_MIN,
            max_value=PLOT_HEIGHT_MAX,
            value=PLOT_HEIGHT_DEFAULT,
            step=1,
            key="plot_height_slider",
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
        st.error(f"Plot error: {e}")
        st.caption(
            "Ensure filtered data has required columns: raman_shift, intensity, "
            "filename, signal_index."
        )
