"""
Legacy Sensor QC tab — CV-style consistency, degradation, batch stability, PDF.
"""

import logging

import streamlit as st

from cache import build_cached_sensor_assessment_artifacts
from components.shared_ui import (
    render_dataframe_stretch,
    render_figure_stretch,
    render_pdf_download_section,
)

from sensd_sers_analysis.application import (
    SensorAssessmentSelection,
    build_sensor_assessment_pdf_bytes,
)
from sensd_sers_analysis.config import BATCH_DEVIATION_Z_THRESHOLD
from sensd_sers_analysis.utils import order_concentration_labels
from sensd_sers_analysis.processing import get_available_feature_columns
from sensd_sers_analysis.visualization import (
    plot_batch_boxplot,
    plot_degradation_trend,
)

logger = logging.getLogger(__name__)


def render(filtered_features, peak_artifacts):
    """
    Render the legacy Sensor QC tab.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe for the current app state.
    peak_artifacts:
        Shared peak artifacts from the derived data bundle.
    """

    feat_cols_avail = get_available_feature_columns(
        filtered_features,
        peak_artifacts.peak_infos_by_serotype,
    )
    has_serotype = "serotype" in filtered_features.columns
    has_conc_group = "concentration_group" in filtered_features.columns

    if not feat_cols_avail:
        st.warning(
            "No feature columns available. Load data with Raman intensity columns "
            "and ensure filters yield samples."
        )
        return

    st.caption(
        "Legacy sensor QC (CV, batch z-scores, degradation). Prefer the "
        "**Sensor assessment** tab for regression-based RMSE/R² and outlier-aware QA."
    )

    if not has_serotype or not has_conc_group:
        st.warning(
            "Assessment requires **serotype** and **concentration_group** columns. "
            "Ensure data is loaded with metadata and preprocess_metadata has run."
        )
        return

    st.markdown(
        "#### Experimental variable control\n"
        "Select a **specific serotype** and **concentration group** before running "
        "assessment. Statistics are computed only on replicates sharing these conditions."
    )
    serotype_opts = sorted(
        filtered_features["serotype"].dropna().unique().astype(str).tolist()
    ) or ["(none)"]
    conc_raw = filtered_features["concentration_group"].dropna().astype(str).unique().tolist()
    conc_opts = [c for c in conc_raw if c and c != "nan"]
    conc_opts = order_concentration_labels(conc_opts) if conc_opts else ["(none)"]

    a_sero, a_conc, a_feat, a_outlier = st.columns(4)
    with a_sero:
        assess_serotype = st.selectbox(
            "Serotype _(required)_",
            options=serotype_opts,
            index=0,
            key="assess_serotype",
        )
    with a_conc:
        assess_concentration = st.selectbox(
            "Concentration group _(required)_",
            options=conc_opts,
            index=0,
            key="assess_concentration",
        )
    with a_feat:
        assess_feature = st.selectbox(
            "Feature",
            options=feat_cols_avail,
            index=0,
            key="assess_feature",
        )
    with a_outlier:
        outlier_method = st.radio(
            "Outlier method",
            options=["iqr", "zscore"],
            index=0,
            horizontal=True,
            key="assess_outlier",
        )

    _sero_valid = assess_serotype and assess_serotype != "(none)"
    _conc_valid = assess_concentration and assess_concentration != "(none)"
    if not _sero_valid or not _conc_valid:
        st.info("Select a specific serotype and concentration group above to run assessment.")
        return

    preview_selection = SensorAssessmentSelection(
        serotype=assess_serotype,
        concentration_group=assess_concentration,
        feature=assess_feature,
        outlier_method=outlier_method,
        batch_feature=assess_feature,
    )
    preview_artifacts = build_cached_sensor_assessment_artifacts(
        filtered_features,
        preview_selection,
    )

    if preview_artifacts.assessment_df.empty:
        st.warning(
            f"No samples for serotype={assess_serotype}, concentration={assess_concentration}. "
            "Adjust filters or selection."
        )
        return

    artifacts = preview_artifacts

    st.markdown("##### Consistency (CV: raw vs. filtered)")
    st.caption(
        f"Within serotype={assess_serotype}, concentration={assess_concentration}. "
        "Grouped by sensor_id, serotype, concentration_group."
    )
    if artifacts.consistency_error:
        logger.warning("Consistency error: %s", artifacts.consistency_error)
        st.error(f"Consistency error: {artifacts.consistency_error}")
    elif not artifacts.display_consistency_table.empty:
        render_dataframe_stretch(artifacts.display_consistency_table)

    st.markdown("##### Degradation trend")
    st.caption(
        "Feature vs. test sequence (test_id or date ordered). Negative slope indicates degradation."
    )
    if artifacts.degradation_error:
        logger.warning("Degradation error: %s", artifacts.degradation_error)
        st.error(f"Degradation error: {artifacts.degradation_error}")
    elif artifacts.degradation_input_df.empty or len(artifacts.degradation_input_df) < 2:
        st.info("Insufficient temporal data (need test_id or date with ≥2 tests).")
    else:
        if not artifacts.degradation_table.empty:
            render_dataframe_stretch(artifacts.degradation_table)
        fig_deg = plot_degradation_trend(
            artifacts.degradation_input_df,
            artifacts.selection.feature,
            "test_ordinal",
            group_col=(
                "sensor_id" if "sensor_id" in artifacts.degradation_input_df.columns else None
            ),
        )
        render_figure_stretch(fig_deg)

    st.markdown("---")
    st.markdown("#### Multi-sensor batch stability")
    st.caption(
        f"Within serotype={assess_serotype}, concentration={assess_concentration}. "
        "Compare sensors under identical conditions."
    )
    if "sensor_id" in preview_artifacts.assessment_df.columns:
        batch_feature = st.selectbox(
            "Feature (batch)",
            options=feat_cols_avail,
            index=feat_cols_avail.index(assess_feature) if assess_feature in feat_cols_avail else 0,
            key="batch_feature",
        )
        selection = SensorAssessmentSelection(
            serotype=assess_serotype,
            concentration_group=assess_concentration,
            feature=assess_feature,
            outlier_method=outlier_method,
            batch_feature=batch_feature,
        )
        artifacts = build_cached_sensor_assessment_artifacts(filtered_features, selection)
        if artifacts.batch_error:
            logger.warning("Batch variance error: %s", artifacts.batch_error)
            st.error(f"Batch variance error: {artifacts.batch_error}")
        else:
            render_dataframe_stretch(artifacts.display_batch_table)
            if not artifacts.display_deviating_sensors_table.empty:
                st.markdown(f"**Deviating sensors (|z| > {BATCH_DEVIATION_Z_THRESHOLD:g})**")
                render_dataframe_stretch(artifacts.display_deviating_sensors_table)

            fig_batch = plot_batch_boxplot(
                artifacts.assessment_df,
                artifacts.display_batch_feature,
                sensor_col="sensor_id",
                group_col=None,
            )
            render_figure_stretch(fig_batch)
    else:
        st.info("No sensor_id column; batch analysis requires sensor identifiers.")

    st.markdown("---")
    st.markdown("#### PDF Report")

    def _generate_assessment_pdf_bytes() -> bytes:
        return build_sensor_assessment_pdf_bytes(artifacts)

    render_pdf_download_section(
        session_key="assessment_pdf",
        filename="sensor_assessment_report.pdf",
        generate_callback=_generate_assessment_pdf_bytes,
        button_label="Generate report",
        download_label="Download PDF",
    )
