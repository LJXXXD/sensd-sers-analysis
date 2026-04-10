"""
Sensor assessment tab — regression QA, global assessment, Phase 1 PDF.
"""

import logging

import streamlit as st

from cache import (
    build_cached_global_qa_artifacts,
    build_cached_single_sensor_consistency_artifacts,
)
from components.shared_ui import (
    render_dataframe_stretch,
    render_figure_stretch,
    render_metrics_row,
    render_pdf_download_section,
)

from sensd_sers_analysis.application import (
    ModelConsistencySelection,
    build_overlay_artifacts,
    build_phase1_pdf_bytes,
)
from sensd_sers_analysis.config import (
    GLOBAL_QA_R2_MIN_THRESHOLD,
    GLOBAL_QA_REJECTION_MULTIPLIER,
)
from sensd_sers_analysis.processing import (
    DEFAULT_GLOBAL_QA_FEATURES,
    get_available_feature_columns,
)
from sensd_sers_analysis.visualization import (
    plot_concentration_regression,
    plot_macro_batch_regression,
    plot_multi_sensor_regression,
)

logger = logging.getLogger(__name__)


def render(filtered_features, peak_artifacts):
    """
    Render the Sensor assessment tab.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe for the current app state.
    peak_artifacts:
        Shared peak artifacts from the derived data bundle.
    """

    mc_feat_cols = get_available_feature_columns(
        filtered_features,
        peak_artifacts.peak_infos_by_serotype,
    )
    has_sensor = "sensor_id" in filtered_features.columns
    has_serotype = "serotype" in filtered_features.columns
    has_log_conc = "log_concentration" in filtered_features.columns

    if not mc_feat_cols:
        st.warning(
            "No feature columns available. Load data with Raman intensity columns "
            "and ensure filters yield samples."
        )
        return
    if not has_sensor or not has_serotype:
        st.warning(
            "Sensor assessment requires **sensor_id** and **serotype** columns. "
            "Ensure data is loaded with metadata and preprocess_metadata has run."
        )
        return
    if not has_log_conc:
        st.warning(
            "Sensor assessment requires **log_concentration**. "
            "Ensure preprocess_metadata has run on the loaded data."
        )
        return

    st.markdown(
        "#### Sensor assessment\n"
        "Two-pass regression with residual-based outlier removal. 0 CFU samples "
        "are excluded from the fit and shown as a horizontal baseline. Outliers "
        "are identified via IQR on absolute residuals and excluded from the "
        "clean fit."
    )
    sensor_opts = sorted(filtered_features["sensor_id"].dropna().unique().astype(str).tolist()) or [
        "(none)"
    ]
    serotype_opts = sorted(
        filtered_features["serotype"].dropna().unique().astype(str).tolist()
    ) or ["(none)"]

    mc_sensor, mc_serotype, mc_feature = st.columns(3)
    with mc_sensor:
        model_sensor = st.selectbox(
            "Sensor ID",
            options=sensor_opts,
            index=0,
            key="sensor_assessment_sensor",
        )
    with mc_serotype:
        model_serotype = st.selectbox(
            "Serotype",
            options=serotype_opts,
            index=0,
            key="sensor_assessment_serotype",
        )
    with mc_feature:
        model_feature = st.selectbox(
            "Feature to assess",
            options=mc_feat_cols,
            index=0,
            key="sensor_assessment_feature",
        )

    _mc_sensor_ok = model_sensor and model_sensor != "(none)"
    _mc_serotype_ok = model_serotype and model_serotype != "(none)"
    if _mc_sensor_ok and _mc_serotype_ok:
        single_selection = ModelConsistencySelection(
            sensor_id=model_sensor,
            serotype=model_serotype,
            feature=model_feature,
        )
        single_artifacts = build_cached_single_sensor_consistency_artifacts(
            filtered_features,
            single_selection,
        )
    else:
        single_artifacts = None

    if (
        single_artifacts is not None
        and single_artifacts.model_df.empty
        and _mc_sensor_ok
        and _mc_serotype_ok
    ):
        st.warning(
            f"No samples for sensor_id={model_sensor}, serotype={model_serotype}. "
            "Adjust filters or selection."
        )
    elif not _mc_sensor_ok or not _mc_serotype_ok:
        st.info("Select a sensor ID and serotype above to run sensor assessment.")
    else:
        assert single_artifacts is not None
        cres = single_artifacts.regression_result
        zero_baseline = single_artifacts.zero_cfu_baseline

        if cres is not None:
            render_metrics_row(
                [
                    ("Raw RMSE", f"{cres.raw_rmse:.4f}"),
                    ("Raw R²", f"{cres.raw_r2:.4f}"),
                    ("Clean RMSE", f"{cres.clean_rmse:.4f}"),
                    ("Clean R²", f"{cres.clean_r2:.4f}"),
                ]
            )
            if cres.n_outliers > 0:
                st.caption(f"Outliers dropped: {cres.n_outliers} (IQR on |residuals|)")
        else:
            st.warning(
                "Insufficient data for regression (need ≥2 samples with valid "
                "log concentration). 0 CFU samples are excluded from the fit."
            )

        try:
            fig_mc = plot_concentration_regression(
                single_artifacts.model_df,
                single_artifacts.selection.feature,
                regression_result=cres.clean_result if cres else None,
                raw_regression_result=cres.raw_result if cres else None,
                zero_cfu_baseline=zero_baseline,
                outlier_mask=cres.outlier_mask if cres else None,
                title=(
                    f"{single_artifacts.selection.sensor_id} — "
                    f"{single_artifacts.selection.serotype}"
                ),
            )
            render_figure_stretch(fig_mc)
        except ValueError as e:
            logger.warning("Model consistency plot error: %s", e)
            st.error(f"Plot error: {e}")

    st.markdown("---")
    st.markdown("#### Global Multi-Sensor Assessment")
    st.caption(
        "Per-sensor QA with dual threshold. Excluded if: Clean RMSE > "
        f"{GLOBAL_QA_REJECTION_MULTIPLIER:g}× batch median OR Clean R² < "
        f"{GLOBAL_QA_R2_MIN_THRESHOLD:.2f} (dead/flat sensor)."
    )

    global_qa_default = [f for f in DEFAULT_GLOBAL_QA_FEATURES if f in mc_feat_cols] or (
        mc_feat_cols[:5] if mc_feat_cols else []
    )
    global_qa_selected = st.multiselect(
        "Features for Global QA Table (and PDF)",
        options=mc_feat_cols,
        default=global_qa_default,
        key="global_qa_features",
    )
    if not global_qa_selected:
        st.info("Select at least one feature to populate the Global QA Table.")
    global_qa_artifacts = build_cached_global_qa_artifacts(
        filtered_features,
        tuple(global_qa_selected),
    )
    if not global_qa_artifacts.table.empty:
        render_dataframe_stretch(
            global_qa_artifacts.table,
            column_config={
                "outliers": st.column_config.NumberColumn("Outliers"),
                "raw_rmse": st.column_config.NumberColumn("Raw RMSE", format="%.4f"),
                "raw_r2": st.column_config.NumberColumn("Raw R²", format="%.4f"),
                "clean_rmse": st.column_config.NumberColumn("Clean RMSE", format="%.4f"),
                "clean_r2": st.column_config.NumberColumn("Clean R²", format="%.4f"),
            },
        )
    else:
        st.info(
            "No regression results. Ensure filtered data has ≥2 valid "
            "(>0 CFU) points per sensor × serotype × feature."
        )

    st.markdown("##### Multi-sensor regression overlay")
    st.caption(
        "Select serotypes and features. Excluded sensors (dashed gray); passing "
        "sensors (solid colors). Tight bundle = uniform batch."
    )
    overlay_sero_opts = [s for s in serotype_opts if s and s != "(none)"]
    overlay_feat_default = (
        ["integral_area"] if "integral_area" in mc_feat_cols else mc_feat_cols[:1]
    )
    overlay_sero, overlay_feat = st.columns(2)
    with overlay_sero:
        overlay_serotypes = st.multiselect(
            "Serotype _(overlay)_",
            options=overlay_sero_opts,
            default=overlay_sero_opts,
            key="overlay_serotype",
        )
    with overlay_feat:
        overlay_features = st.multiselect(
            "Feature _(overlay)_",
            options=mc_feat_cols,
            default=overlay_feat_default,
            key="overlay_feature",
        )

    overlay_artifacts = build_overlay_artifacts(
        filtered_features,
        tuple(overlay_serotypes),
        tuple(overlay_features),
        global_qa_artifacts.excluded_map,
    )
    for artifact in overlay_artifacts:
        st.markdown(f"**{artifact.serotype} — {artifact.feature}**")
        try:
            fig_ov = plot_multi_sensor_regression(
                filtered_features,
                artifact.serotype,
                artifact.feature,
                excluded_sensors=set(artifact.excluded_sensors),
            )
            render_figure_stretch(fig_ov)
        except ValueError as e:
            logger.warning(
                "Overlay plot error (%s, %s): %s",
                artifact.serotype,
                artifact.feature,
                e,
            )
            st.error(f"Overlay ({artifact.serotype}, {artifact.feature}): {e}")

        st.markdown("**Macro batch regression**")
        try:
            fig_macro, macro_res = plot_macro_batch_regression(
                filtered_features,
                artifact.serotype,
                artifact.feature,
                set(artifact.pass_sensors),
            )
            render_figure_stretch(fig_macro)
            if macro_res is not None:
                render_metrics_row(
                    [
                        ("Raw Batch RMSE", f"{macro_res.raw_batch_rmse:.4f}"),
                        ("Raw Batch R²", f"{macro_res.raw_batch_r2:.4f}"),
                        ("Clean Batch RMSE", f"{macro_res.clean_batch_rmse:.4f}"),
                        ("Clean Batch R²", f"{macro_res.clean_batch_r2:.4f}"),
                        ("Macro Outliers", f"{macro_res.n_macro_outliers}"),
                    ]
                )
        except ValueError as e:
            logger.warning(
                "Macro batch regression error (%s, %s): %s",
                artifact.serotype,
                artifact.feature,
                e,
            )
            st.error(f"Macro ({artifact.serotype}, {artifact.feature}): {e}")
        st.markdown("---")

    if not overlay_serotypes or not overlay_features:
        st.info("Select at least one serotype and one feature to generate plots.")
        st.markdown("---")
    st.markdown("#### PDF Report")

    def _generate_phase1_pdf_bytes() -> bytes:
        return build_phase1_pdf_bytes(
            filtered_features,
            global_qa_artifacts,
            overlay_artifacts,
            report_title="Sensor Consistency & Quality Assurance Report",
        )

    render_pdf_download_section(
        session_key="phase1_qa_pdf",
        filename="sensor_assessment_regression_report.pdf",
        generate_callback=_generate_phase1_pdf_bytes,
        button_label="Generate Sensor Assessment Report",
        download_label="Download Sensor Assessment Report",
    )
