"""
Sensor Assessment & Report tab — consistency, degradation, batch stability, PDF.
"""

import streamlit as st

from sensd_sers_analysis.assessment import (
    compute_batch_variance,
    compute_degradation,
    get_consistency_summary_table,
    identify_deviating_sensors,
    prepare_degradation_data,
)
from sensd_sers_analysis.report import build_sensor_assessment_pdf
from sensd_sers_analysis.utils import order_concentration_labels
from sensd_sers_analysis.visualization import (
    plot_batch_boxplot,
    plot_degradation_trend,
)

from utils import get_available_feature_columns

ASSESSMENT_GROUP_COLS = ["sensor_id", "serotype", "concentration_group"]


def render(filtered_features):
    """Render the Sensor Assessment & Report tab."""
    feat_cols_avail = get_available_feature_columns(
        filtered_features,
        st.session_state.get("peak_infos_by_serotype", {}),
    )
    has_serotype = "serotype" in filtered_features.columns
    has_conc_group = "concentration_group" in filtered_features.columns

    if not feat_cols_avail:
        st.warning(
            "No feature columns available. Load data with Raman intensity columns "
            "and ensure filters yield samples."
        )
        return
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
    conc_raw = (
        filtered_features["concentration_group"].dropna().astype(str).unique().tolist()
    )
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
    if _sero_valid and _conc_valid:
        assessment_df = filtered_features[
            (filtered_features["serotype"].astype(str) == assess_serotype)
            & (
                filtered_features["concentration_group"].astype(str)
                == assess_concentration
            )
        ].copy()
    else:
        assessment_df = filtered_features.iloc[0:0].copy()

    if assessment_df.empty and (_sero_valid and _conc_valid):
        st.warning(
            f"No samples for serotype={assess_serotype}, concentration={assess_concentration}. "
            "Adjust filters or selection."
        )
        return
    if not _sero_valid or not _conc_valid:
        st.info(
            "Select a specific serotype and concentration group above to run assessment."
        )
        return

    consistency_group_cols = [
        c for c in ASSESSMENT_GROUP_COLS if c in assessment_df.columns
    ]
    if not consistency_group_cols:
        consistency_group_cols = (
            ["sensor_id"] if "sensor_id" in assessment_df.columns else None
        )

    st.markdown("##### Consistency (CV: raw vs. filtered)")
    st.caption(
        f"Within serotype={assess_serotype}, concentration={assess_concentration}. "
        "Grouped by sensor_id, serotype, concentration_group."
    )
    try:
        consistency_tbl = get_consistency_summary_table(
            assessment_df,
            feature_cols=[assess_feature],
            group_cols=consistency_group_cols,
            outlier_method=outlier_method,
        )
        if not consistency_tbl.empty:
            st.dataframe(consistency_tbl, width="stretch", hide_index=True)
    except ValueError as e:
        st.error(f"Consistency error: {e}")

    st.markdown("##### Degradation trend")
    st.caption(
        "Feature vs. test sequence (test_id or date ordered). "
        "Negative slope indicates degradation."
    )
    try:
        df_deg = prepare_degradation_data(
            assessment_df, assess_feature, test_col="test_id", date_col="date"
        )
        if df_deg.empty or len(df_deg) < 2:
            st.info("Insufficient temporal data (need test_id or date with ≥2 tests).")
        else:
            deg_tbl = compute_degradation(
                df_deg,
                assess_feature,
                "test_ordinal",
                group_cols=["sensor_id"] if "sensor_id" in df_deg.columns else None,
            )
            if not deg_tbl.empty:
                st.dataframe(deg_tbl, width="stretch", hide_index=True)

            fig_deg = plot_degradation_trend(
                df_deg,
                assess_feature,
                "test_ordinal",
                group_col="sensor_id" if "sensor_id" in df_deg.columns else None,
            )
            st.pyplot(fig_deg, width="stretch")
    except ValueError as e:
        st.error(f"Degradation error: {e}")

    st.markdown("---")
    st.markdown("#### Multi-sensor batch stability")
    st.caption(
        f"Within serotype={assess_serotype}, concentration={assess_concentration}. "
        "Compare sensors under identical conditions."
    )
    if "sensor_id" in assessment_df.columns:
        batch_feature = st.selectbox(
            "Feature (batch)",
            options=feat_cols_avail,
            index=feat_cols_avail.index(assess_feature)
            if assess_feature in feat_cols_avail
            else 0,
            key="batch_feature",
        )
        try:
            batch_tbl = compute_batch_variance(
                assessment_df,
                batch_feature,
                sensor_col="sensor_id",
                group_cols=None,
            )
            deviating = identify_deviating_sensors(
                batch_tbl, z_threshold=2.0, sensor_col="sensor_id"
            )

            st.dataframe(batch_tbl, width="stretch", hide_index=True)
            if not deviating.empty:
                st.markdown("**Deviating sensors (|z| > 2)**")
                st.dataframe(deviating, width="stretch", hide_index=True)

            fig_batch = plot_batch_boxplot(
                assessment_df,
                batch_feature,
                sensor_col="sensor_id",
                group_col=None,
            )
            st.pyplot(fig_batch, width="stretch")
        except ValueError as e:
            st.error(f"Batch variance error: {e}")
    else:
        st.info("No sensor_id column; batch analysis requires sensor identifiers.")

    st.markdown("---")
    st.markdown("#### PDF report")
    if st.button("Generate report", key="pdf_report_btn"):
        try:
            _generate_assessment_pdf(
                assessment_df,
                assess_feature,
                assess_serotype,
                assess_concentration,
                consistency_group_cols,
                outlier_method,
            )
            st.success("Report generated. Click Download below.")
        except Exception as e:
            st.error(f"Report generation failed: {e}")

    if "assessment_pdf" in st.session_state:
        st.download_button(
            label="Download PDF",
            data=st.session_state["assessment_pdf"],
            file_name="sensor_assessment_report.pdf",
            mime="application/pdf",
            key="pdf_download",
        )


def _generate_assessment_pdf(
    assessment_df,
    assess_feature,
    assess_serotype,
    assess_concentration,
    consistency_group_cols,
    outlier_method,
):
    """Build and store sensor assessment PDF in session state."""
    consistency_summary = get_consistency_summary_table(
        assessment_df,
        group_cols=consistency_group_cols,
        outlier_method=outlier_method,
    )
    df_deg_pdf = prepare_degradation_data(
        assessment_df,
        assess_feature,
        test_col="test_id",
        date_col="date",
    )
    deg_summary = None
    fig_deg_pdf = None
    if not df_deg_pdf.empty and len(df_deg_pdf) >= 2:
        deg_summary = compute_degradation(
            df_deg_pdf,
            assess_feature,
            "test_ordinal",
            group_cols=(["sensor_id"] if "sensor_id" in df_deg_pdf.columns else None),
        )
        fig_deg_pdf = plot_degradation_trend(
            df_deg_pdf,
            assess_feature,
            "test_ordinal",
            group_col=("sensor_id" if "sensor_id" in df_deg_pdf.columns else None),
        )

    batch_tbl_pdf = None
    fig_batch_pdf = None
    deviating_pdf = None
    if "sensor_id" in assessment_df.columns:
        batch_tbl_pdf = compute_batch_variance(
            assessment_df,
            assess_feature,
            sensor_col="sensor_id",
            group_cols=None,
        )
        deviating_pdf = identify_deviating_sensors(
            batch_tbl_pdf, z_threshold=2.0, sensor_col="sensor_id"
        )
        fig_batch_pdf = plot_batch_boxplot(
            assessment_df,
            assess_feature,
            sensor_col="sensor_id",
            group_col=None,
        )

    pdf_bytes = build_sensor_assessment_pdf(
        consistency_table=consistency_summary,
        degradation_table=deg_summary,
        degradation_fig=fig_deg_pdf,
        batch_variance_table=(
            batch_tbl_pdf
            if batch_tbl_pdf is not None and not batch_tbl_pdf.empty
            else None
        ),
        batch_boxplot_fig=fig_batch_pdf,
        deviating_sensors_table=(
            deviating_pdf
            if deviating_pdf is not None and not deviating_pdf.empty
            else None
        ),
        outlier_method=outlier_method,
        report_title=(
            f"SERS Sensor Assessment — {assess_serotype}, {assess_concentration}"
        ),
    )
    st.session_state["assessment_pdf"] = pdf_bytes
