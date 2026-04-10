"""
Peak Feature Extraction tab — fixed-anchor peak heights for downstream analysis.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

from components.shared_ui import render_figure_stretch
from sensd_sers_analysis.application import (
    build_matching_signal_options,
    build_peak_diagnostic_context,
    build_signal_selection_options,
    merge_targeted_peaks_into_filtered_bundle,
)
from sensd_sers_analysis.config.targeted_peaks import (
    TARGETED_PEAK_DEFAULT_ANCHORS_CM1,
    TARGETED_PEAK_DEFAULT_COUNT,
    TARGETED_PEAK_SEARCH_HALF_WIDTH_CM1,
)
from sensd_sers_analysis.data import get_raman_shift, get_signals_matrix
from sensd_sers_analysis.processing.targeted_peak_features import (
    compute_targeted_peak_positions_on_mean,
    detect_targeted_peaks_on_spectrum_row,
    list_targeted_peak_feature_columns,
    target_anchor_to_feature_name,
)
from sensd_sers_analysis.visualization import (
    plot_targeted_mean_spectrum_markers,
    plot_targeted_signal_verification,
)
from theme import (
    AXVLINE_ALPHA,
    DEFAULT_FIGSIZE_ANCHOR,
    DEFAULT_FIGSIZE_WIDE,
    GRID_ALPHA,
    LEGEND_FONTSIZE,
)

logger = logging.getLogger(__name__)


def render(filtered_bundle, derived_bundle, peak_artifacts) -> None:
    """
    Render fixed-anchor peak extraction controls, serotype means, and signal QA.

    Parameters
    ----------
    filtered_bundle:
        Filtered bundle (base features; targeted columns are merged for QA).
    derived_bundle:
        Derived bundle (trimmed wide spectra for recomputation on the fly).
    peak_artifacts:
        Dynamic peak artifacts for diagnostic context alignment.
    """

    st.markdown(
        "Define Raman-shift anchors (cm⁻¹). For each spectrum, the pipeline "
        "records a **baseline-adjusted height** at the maximum inside a "
        f"±{TARGETED_PEAK_SEARCH_HALF_WIDTH_CM1:.1f} cm⁻¹ window around each "
        "anchor. Column names encode the target (for example "
        "``peak_near_501_8`` for 501.8 cm⁻¹). These columns are joined into "
        "the feature matrix used by Feature Analysis, legacy Sensor QC, "
        "Sensor assessment, and Serotype Classification."
    )

    defaults = list(TARGETED_PEAK_DEFAULT_ANCHORS_CM1)
    n_pf = st.number_input(
        "Number of peak features",
        min_value=1,
        max_value=20,
        value=TARGETED_PEAK_DEFAULT_COUNT,
        step=1,
        key="pfe_n_features",
        help="How many targeted peak-height columns to extract per spectrum.",
    )

    anchors: list[float] = []
    st.markdown("##### Target Raman shifts (cm⁻¹)")
    for row_start in range(0, int(n_pf), 3):
        row_cols = st.columns(3)
        for j in range(3):
            idx = row_start + j
            if idx >= int(n_pf):
                break
            default_val = float(defaults[idx]) if idx < len(defaults) else float(defaults[-1])
            with row_cols[j]:
                anchors.append(
                    float(
                        st.number_input(
                            f"Anchor {idx + 1}",
                            value=default_val,
                            format="%.4f",
                            key=f"pfe_anchor_cm1_{idx}",
                        )
                    )
                )

    st.session_state["pfe_targeted_anchors"] = tuple(anchors)

    merged_preview = merge_targeted_peaks_into_filtered_bundle(
        filtered_bundle,
        derived_bundle.wide_df,
        tuple(anchors),
    )
    peak_cols = list_targeted_peak_feature_columns(merged_preview.filtered_features_df.columns)
    if peak_cols:
        st.caption("Active feature columns: " + ", ".join(f"``{c}``" for c in peak_cols))

    feat = filtered_bundle.filtered_features_df
    wide = derived_bundle.wide_df
    if feat.empty or wide.empty:
        st.warning("No filtered samples or wide spectra available.")
        return

    st.markdown("##### Serotype mean spectra (targeted peaks)")
    if "serotype" not in feat.columns:
        st.caption("Mean-by-serotype plots require a **serotype** column.")
    else:
        for serotype in sorted(feat["serotype"].dropna().astype(str).unique()):
            rows = feat[feat["serotype"].astype(str) == serotype]
            idx = rows.index
            wsub = wide.reindex(idx)
            if wsub.empty:
                continue
            signals = get_signals_matrix(wsub)
            if signals.size == 0 or not np.any(np.isfinite(signals)):
                continue
            with st.expander(f"**{serotype}** — mean spectrum & targeted peaks", expanded=True):
                mean_spec = np.nanmean(signals, axis=0)
                raman_x = np.asarray(get_raman_shift(wsub), dtype=float)
                det_shift, det_raw, _det_adj = compute_targeted_peak_positions_on_mean(
                    mean_spec,
                    raman_x,
                    anchors,
                )
                fig_m = plot_targeted_mean_spectrum_markers(
                    raman_x,
                    mean_spec,
                    anchors,
                    det_shift,
                    det_raw,
                    serotype,
                    figsize=DEFAULT_FIGSIZE_ANCHOR,
                    legend_fontsize=LEGEND_FONTSIZE,
                    grid_alpha=GRID_ALPHA,
                    axvline_alpha=AXVLINE_ALPHA,
                )
                render_figure_stretch(fig_m)

    st.markdown("#### Signal-level verification")
    st.caption(
        "Inspect one spectrum: dashed lines = anchor targets; green ★ = local "
        "maximum inside each search window. The table lists baseline-adjusted "
        "heights stored in the feature matrix."
    )

    try:
        context = build_peak_diagnostic_context(
            filtered_bundle.filtered_features_df,
            derived_bundle.wide_df,
            peak_artifacts,
            require_non_empty_peak_artifacts=False,
        )
    except KeyError as exc:
        logger.warning("Peak feature extraction context error: %s", exc)
        st.error(f"Verification context error: {exc}")
        return

    if context is None or context.wide_filtered.empty:
        st.info(
            "Verification needs aligned wide spectra and metadata "
            "(sensor_id, concentration_group, serotype when present)."
        )
        return

    if not (context.sensor_col and context.concentration_col):
        st.caption("Signal-level verification requires **sensor_id** and **concentration_group**.")
        return

    serotypes = sorted(feat["serotype"].dropna().astype(str).unique()) if "serotype" in feat else []
    if not serotypes:
        st.info("Add **serotype** to filter signals for verification.")
        return

    selected_serotype = st.selectbox(
        "Serotype (filter signals)",
        options=serotypes,
        index=0,
        key="pfe_serotype_filter",
    )
    signal_options = build_signal_selection_options(context, selected_serotype)
    if not signal_options.sensor_options or not signal_options.concentration_options:
        st.info(f"No signals for serotype **{selected_serotype}** under current filters.")
        return

    col_sensor, col_concentration, col_signal = st.columns(3)
    with col_sensor:
        selected_sensor = st.selectbox(
            "Sensor ID",
            options=signal_options.sensor_options,
            index=0,
            key="pfe_sensor",
        )
    with col_concentration:
        selected_concentration = st.selectbox(
            "Concentration",
            options=signal_options.concentration_options,
            index=0,
            key="pfe_conc",
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
                key="pfe_signal",
            )
        else:
            signal_position = 0

    if not matching_options.row_indices:
        st.warning("No matching signal for the selected sensor and concentration.")
        return

    row_idx = matching_options.row_indices[
        min(int(signal_position), len(matching_options.row_indices) - 1)
    ]
    spec_row = context.wide_filtered.loc[[row_idx]]
    sig_mat = get_signals_matrix(spec_row)
    rx = np.asarray(get_raman_shift(spec_row), dtype=float)
    y_row = sig_mat[0]
    valid = np.isfinite(y_row.astype(float))
    x_plot = rx[valid]
    y_plot = np.asarray(y_row, dtype=float)[valid]
    order = np.argsort(x_plot)
    x_plot = x_plot[order]
    y_plot = y_plot[order]

    det_shifts, det_raw, det_adj = detect_targeted_peaks_on_spectrum_row(
        np.asarray(y_row, dtype=float),
        rx,
        anchors,
    )
    title = (
        f"Signal: {selected_sensor} @ {selected_concentration} "
        f"({selected_serotype}) | Targeted peaks"
    )
    fig_s = plot_targeted_signal_verification(
        x_plot,
        y_plot,
        anchors,
        det_shifts,
        det_raw,
        title=title,
        figsize=DEFAULT_FIGSIZE_WIDE,
        legend_fontsize=LEGEND_FONTSIZE,
        grid_alpha=GRID_ALPHA,
        axvline_alpha=AXVLINE_ALPHA,
    )
    render_figure_stretch(fig_s)

    row_features = merged_preview.filtered_features_df.loc[row_idx]
    value_map: dict[str, float] = {}
    for anchor in anchors:
        col_name = target_anchor_to_feature_name(float(anchor))
        if col_name in merged_preview.filtered_features_df.columns:
            raw_val = row_features[col_name]
            value_map[col_name] = float(raw_val) if pd.notna(raw_val) else float("nan")
    if value_map:
        st.markdown("**Stored feature values (baseline-adjusted heights)**")
        st.dataframe(
            data=[value_map],
            use_container_width=True,
        )
