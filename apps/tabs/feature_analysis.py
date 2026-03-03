"""
Feature Analysis tab — feature distribution plots.
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
    BASIC_FEATURE_COLUMNS,
    get_feature_metadata_columns,
    pick_preferred_column,
)
from sensd_sers_analysis.processing import get_available_feature_columns
from sensd_sers_analysis.visualization import plot_feature_distribution


def render(filtered_features):
    """Render the Feature Analysis tab."""
    all_feat_nan = all(
        filtered_features[c].isna().all()
        for c in BASIC_FEATURE_COLUMNS
        if c in filtered_features.columns
    )
    if all_feat_nan:
        st.warning(
            "All extracted features are NaN. This usually means the loaded data "
            "lacks Raman intensity columns (rs_*) or they contain no valid "
            "numeric values. Ensure your Excel files use the expected embedded "
            "format with Raman shift columns."
        )
        return

    st.markdown("#### Feature distribution options")
    x_opts = get_feature_metadata_columns(filtered_features)
    hue_opts = ["None"] + x_opts

    x_default = pick_preferred_column(x_opts) or (x_opts[0] if x_opts else None)
    x_default_idx = x_opts.index(x_default) if x_default in x_opts else 0

    stats_feat_opts = get_available_feature_columns(
        filtered_features,
        st.session_state.get("peak_infos_by_serotype", {}),
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        feature_col = st.selectbox(
            "Feature (Y-axis)",
            options=stats_feat_opts if stats_feat_opts else ["(no features)"],
            index=min(2, len(stats_feat_opts) - 1) if stats_feat_opts else 0,
            key="stats_feature",
        )
    with col2:
        x_col = st.selectbox(
            "X-axis",
            options=x_opts if x_opts else ["(no grouping)"],
            index=min(x_default_idx, len(x_opts) - 1) if x_opts else 0,
            key="stats_x",
        )
    with col3:
        hue_default_s = pick_preferred_column(x_opts, ("serotype",)) or "None"
        hue_col_s = st.selectbox(
            "Hue",
            options=hue_opts,
            index=hue_opts.index(hue_default_s),
            key="stats_hue",
        )
    col_plot_type, col_height = st.columns(2)
    with col_plot_type:
        plot_type = st.radio(
            "Plot type",
            options=["box", "violin"],
            index=0,
            horizontal=True,
            key="stats_plot_type",
        )
    with col_height:
        plot_height = st.slider(
            "Height (in)",
            min_value=PLOT_HEIGHT_MIN,
            max_value=PLOT_HEIGHT_MAX,
            value=PLOT_HEIGHT_DEFAULT,
            step=1,
            key="stats_plot_height",
        )

    if filtered_features.empty:
        st.warning("No samples match the selected filters for feature analysis.")
        return

    x_val = (
        None
        if (
            not x_col
            or x_col == "(no grouping)"
            or x_col not in filtered_features.columns
        )
        else x_col
    )
    hue_val = None if hue_col_s == "None" else hue_col_s
    try:
        fig_stats = plot_feature_distribution(
            filtered_features,
            feature_col,
            x=x_val,
            hue=hue_val,
            plot_type=plot_type,
            figsize=(DEFAULT_FIGSIZE_WIDTH, plot_height),
        )
        render_figure_stretch(fig_stats)
    except ValueError as e:
        st.error(f"Plot error: {e}")
