"""
Peak Detection Diagnostics tab — serotype-specific peak verification.
"""

import logging

import streamlit as st

from components.shared_ui import render_dataframe_stretch, render_figure_stretch
from sensd_sers_analysis.application import (
    build_matching_signal_options,
    build_peak_anchor_overviews,
    build_peak_anchor_table,
    build_peak_diagnostic_context,
    build_signal_selection_options,
    build_signal_verification_artifact,
)
from sensd_sers_analysis.visualization import (
    plot_peak_anchor_summary,
    plot_signal_level_peak_verification,
)
from theme import (
    AXVLINE_ALPHA,
    DEFAULT_FIGSIZE_ANCHOR,
    DEFAULT_FIGSIZE_WIDE,
    GRID_ALPHA,
    LEGEND_FONTSIZE,
    SPAN_ALPHA_ANCHOR,
    SPAN_ALPHA_SIGNAL,
)

logger = logging.getLogger(__name__)


def render(filtered_features, wide_df, peak_artifacts):
    """
    Render the Peak Detection Diagnostics tab.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe for the current app state.
    wide_df:
        Wide dataframe from the derived data bundle.
    peak_artifacts:
        Shared peak artifacts from dynamic peak extraction.
    """

    try:
        context = build_peak_diagnostic_context(filtered_features, wide_df, peak_artifacts)
    except KeyError as exc:
        logger.warning("Peak diagnostics index alignment error: %s", exc)
        st.error(f"Peak diagnostics error: {exc}")
        return

    if context is None:
        logger.info("Peak diagnostics skipped: no peak data (peak_artifacts empty)")
        st.info(
            "Peak detection requires loaded data with Raman intensity columns. "
            "Adjust **Peaks per serotype** in the sidebar and ensure high-concentration "
            "samples are present (>0 CFU, serotype-specific)."
        )
        return

    st.markdown(
        "Visual verification of dynamic peak extraction: serotype-specific "
        "anchors, search windows, and detection success rates (0 CFU excluded from "
        "learning). Each serotype uses its own peak count from the sidebar."
    )

    for overview in build_peak_anchor_overviews(peak_artifacts):
        with st.expander(
            f"**{overview.serotype}** — Mean spectrum & diagnostics",
            expanded=True,
        ):
            fig_anchor = plot_peak_anchor_summary(
                peak_artifacts.raman_x,
                overview.mean_spectrum,
                overview.peak_infos,
                overview.serotype,
                figsize=DEFAULT_FIGSIZE_ANCHOR,
                legend_fontsize=LEGEND_FONTSIZE,
                grid_alpha=GRID_ALPHA,
                span_alpha=SPAN_ALPHA_ANCHOR,
                axvline_alpha=AXVLINE_ALPHA,
            )
            render_figure_stretch(fig_anchor)
            render_dataframe_stretch(build_peak_anchor_table(overview.peak_infos))

    st.markdown("#### Signal-level verification")
    st.caption(
        "Inspect a single spectrum: shaded regions = serotype-specific search "
        "windows; green ★ = where the algorithm detected a peak (passed prominence "
        "check). Pick a serotype below to filter signals — this ties verification "
        "to the plots above."
    )
    if not (context.sensor_col and context.concentration_col and not context.wide_filtered.empty):
        st.caption(
            "Signal-level verification requires sensor_id and concentration_group in the data."
        )
        return

    serotype_options = sorted(peak_artifacts.peak_infos_by_serotype.keys())
    selected_serotype = st.selectbox(
        "Serotype (filter signals)",
        options=serotype_options,
        index=0,
        key="peak_diag_serotype_filter",
        help="Only show sensors/concentrations that have signals of this serotype.",
    )
    signal_options = build_signal_selection_options(context, selected_serotype)
    if not signal_options.sensor_options or not signal_options.concentration_options:
        st.info(
            f"No signals for serotype **{selected_serotype}** in the current filters. "
            "Adjust filters or select another serotype."
        )
        return

    col_sensor, col_concentration, col_signal = st.columns(3)
    with col_sensor:
        selected_sensor = st.selectbox(
            "Sensor ID",
            options=signal_options.sensor_options,
            index=0,
            key="peak_diag_sensor",
        )
    with col_concentration:
        selected_concentration = st.selectbox(
            "Concentration",
            options=signal_options.concentration_options,
            index=0,
            key="peak_diag_conc",
        )

    matching_options = build_matching_signal_options(
        context,
        selected_serotype=selected_serotype,
        selected_sensor=selected_sensor,
        selected_concentration=selected_concentration,
    )
    with col_signal:
        if len(matching_options.signal_labels) > 1:
            signal_position = st.selectbox(
                "Signal",
                options=range(len(matching_options.signal_labels)),
                format_func=lambda i: matching_options.signal_labels[i],
                key="peak_diag_signal",
            )
        else:
            signal_position = 0

    if not matching_options.row_indices:
        st.warning("No matching signal for selected sensor and concentration.")
        return

    verification_artifact = build_signal_verification_artifact(
        context,
        selected_serotype=selected_serotype,
        selected_sensor=selected_sensor,
        selected_concentration=selected_concentration,
        signal_position=signal_position,
    )
    if verification_artifact is None:
        st.warning("No matching signal for selected sensor and concentration.")
        return

    fig_signal = plot_signal_level_peak_verification(
        verification_artifact,
        figsize=DEFAULT_FIGSIZE_WIDE,
        legend_fontsize=LEGEND_FONTSIZE,
        grid_alpha=GRID_ALPHA,
        span_alpha=SPAN_ALPHA_SIGNAL,
    )
    render_figure_stretch(fig_signal)
