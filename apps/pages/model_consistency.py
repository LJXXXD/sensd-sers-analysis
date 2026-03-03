"""
Model-Based Sensor Consistency tab — regression QA, global assessment, Phase 1 PDF.
"""

import streamlit as st

from sensd_sers_analysis.assessment import (
    fit_concentration_regression_cleaned,
    get_global_model_consistency_qa,
    get_zero_cfu_baseline,
)
from sensd_sers_analysis.processing import DEFAULT_GLOBAL_QA_FEATURES
from sensd_sers_analysis.report import build_phase1_qa_pdf
from sensd_sers_analysis.visualization import (
    plot_concentration_regression,
    plot_macro_batch_regression,
    plot_multi_sensor_regression,
)

from utils import get_available_feature_columns


def render(filtered_features):
    """Render the Model-Based Sensor Consistency tab."""
    mc_feat_cols = get_available_feature_columns(
        filtered_features,
        st.session_state.get("peak_infos_by_serotype", {}),
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
            "Model-Based Consistency requires **sensor_id** and **serotype** columns. "
            "Ensure data is loaded with metadata and preprocess_metadata has run."
        )
        return
    if not has_log_conc:
        st.warning(
            "Model-Based Consistency requires **log_concentration**. "
            "Ensure preprocess_metadata has run on the loaded data."
        )
        return

    st.markdown(
        "#### Model-based sensor consistency\n"
        "Two-pass regression with residual-based outlier removal. 0 CFU samples "
        "are excluded from the fit and shown as a horizontal baseline. Outliers "
        "are identified via IQR on absolute residuals and excluded from the "
        "clean fit."
    )
    sensor_opts = sorted(
        filtered_features["sensor_id"].dropna().unique().astype(str).tolist()
    ) or ["(none)"]
    serotype_opts = sorted(
        filtered_features["serotype"].dropna().unique().astype(str).tolist()
    ) or ["(none)"]

    mc_sensor, mc_serotype, mc_feature = st.columns(3)
    with mc_sensor:
        model_sensor = st.selectbox(
            "Sensor ID",
            options=sensor_opts,
            index=0,
            key="model_consistency_sensor",
        )
    with mc_serotype:
        model_serotype = st.selectbox(
            "Serotype",
            options=serotype_opts,
            index=0,
            key="model_consistency_serotype",
        )
    with mc_feature:
        model_feature = st.selectbox(
            "Feature to assess",
            options=mc_feat_cols,
            index=0,
            key="model_consistency_feature",
        )

    _mc_sensor_ok = model_sensor and model_sensor != "(none)"
    _mc_serotype_ok = model_serotype and model_serotype != "(none)"
    if _mc_sensor_ok and _mc_serotype_ok:
        model_df = filtered_features[
            (filtered_features["sensor_id"].astype(str) == model_sensor)
            & (filtered_features["serotype"].astype(str) == model_serotype)
        ].copy()
    else:
        model_df = filtered_features.iloc[0:0].copy()

    if model_df.empty and (_mc_sensor_ok and _mc_serotype_ok):
        st.warning(
            f"No samples for sensor_id={model_sensor}, serotype={model_serotype}. "
            "Adjust filters or selection."
        )
    elif not _mc_sensor_ok or not _mc_serotype_ok:
        st.info("Select a sensor ID and serotype above to run model-based consistency.")
    else:
        cres = fit_concentration_regression_cleaned(model_df, model_feature)
        zero_baseline = get_zero_cfu_baseline(model_df, model_feature)

        if cres is not None:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Raw RMSE", f"{cres.raw_rmse:.4f}")
            with m2:
                st.metric("Raw R²", f"{cres.raw_r2:.4f}")
            with m3:
                st.metric("Clean RMSE", f"{cres.clean_rmse:.4f}")
            with m4:
                st.metric("Clean R²", f"{cres.clean_r2:.4f}")
            if cres.n_outliers > 0:
                st.caption(f"Outliers dropped: {cres.n_outliers} (IQR on |residuals|)")
        else:
            st.warning(
                "Insufficient data for regression (need ≥2 samples with valid "
                "log concentration). 0 CFU samples are excluded from the fit."
            )

        try:
            fig_mc = plot_concentration_regression(
                model_df,
                model_feature,
                regression_result=cres.clean_result if cres else None,
                raw_regression_result=cres.raw_result if cres else None,
                zero_cfu_baseline=zero_baseline,
                outlier_mask=cres.outlier_mask if cres else None,
                title=f"{model_sensor} — {model_serotype}",
            )
            st.pyplot(fig_mc, width="stretch")
        except ValueError as e:
            st.error(f"Plot error: {e}")

    st.markdown("---")
    st.markdown("#### Global Multi-Sensor Assessment")
    st.caption(
        "Per-sensor QA with dual threshold. Excluded if: Clean RMSE > 2× batch "
        "median OR Clean R² < 0.80 (dead/flat sensor)."
    )

    global_qa_default = [
        f for f in DEFAULT_GLOBAL_QA_FEATURES if f in mc_feat_cols
    ] or (mc_feat_cols[:5] if mc_feat_cols else [])
    global_qa_selected = st.multiselect(
        "Features for Global QA Table (and PDF)",
        options=mc_feat_cols,
        default=global_qa_default,
        key="global_qa_features",
    )
    if not global_qa_selected:
        st.info("Select at least one feature to populate the Global QA Table.")
    global_qa_tbl, excluded_map = get_global_model_consistency_qa(
        filtered_features,
        feature_cols=global_qa_selected,
    )
    if not global_qa_tbl.empty:
        st.dataframe(
            global_qa_tbl,
            width="stretch",
            hide_index=True,
            column_config={
                "outliers": st.column_config.NumberColumn("Outliers"),
                "raw_rmse": st.column_config.NumberColumn("Raw RMSE", format="%.4f"),
                "raw_r2": st.column_config.NumberColumn("Raw R²", format="%.4f"),
                "clean_rmse": st.column_config.NumberColumn(
                    "Clean RMSE", format="%.4f"
                ),
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

    overlay_items: list = []
    macro_items: list = []
    for sero in overlay_serotypes:
        for feat in overlay_features:
            excluded = excluded_map.get((str(sero), str(feat)), set())
            all_sens = set(
                str(s)
                for s in filtered_features[
                    filtered_features["serotype"].astype(str) == sero
                ]["sensor_id"]
                .dropna()
                .unique()
            )
            pass_sens = all_sens - excluded

            st.markdown(f"**{sero} — {feat}**")
            try:
                fig_ov = plot_multi_sensor_regression(
                    filtered_features,
                    sero,
                    feat,
                    excluded_sensors=excluded,
                )
                st.pyplot(fig_ov, width="stretch")
                overlay_items.append({"fig": fig_ov, "serotype": sero, "feature": feat})
            except ValueError as e:
                st.error(f"Overlay ({sero}, {feat}): {e}")

            st.markdown("**Macro batch regression**")
            try:
                fig_macro, macro_res = plot_macro_batch_regression(
                    filtered_features,
                    sero,
                    feat,
                    pass_sens,
                )
                st.pyplot(fig_macro, width="stretch")
                macro_items.append(
                    {
                        "fig": fig_macro,
                        "macro_result": macro_res,
                        "serotype": sero,
                        "feature": feat,
                    }
                )
                if macro_res is not None:
                    ma1, ma2, ma3, ma4, ma5 = st.columns(5)
                    with ma1:
                        st.metric(
                            "Raw Batch RMSE",
                            f"{macro_res.raw_batch_rmse:.4f}",
                        )
                    with ma2:
                        st.metric(
                            "Raw Batch R²",
                            f"{macro_res.raw_batch_r2:.4f}",
                        )
                    with ma3:
                        st.metric(
                            "Clean Batch RMSE",
                            f"{macro_res.clean_batch_rmse:.4f}",
                        )
                    with ma4:
                        st.metric(
                            "Clean Batch R²",
                            f"{macro_res.clean_batch_r2:.4f}",
                        )
                    with ma5:
                        st.metric(
                            "Macro Outliers",
                            f"{macro_res.n_macro_outliers}",
                        )
            except ValueError as e:
                st.error(f"Macro ({sero}, {feat}): {e}")
            st.markdown("---")

    if not overlay_serotypes or not overlay_features:
        st.info("Select at least one serotype and one feature to generate plots.")

    st.markdown("---")
    st.markdown("#### Phase 1 QA Report")
    if st.button("Generate Phase 1 QA Report", key="phase1_qa_pdf_btn"):
        try:
            pdf_bytes = build_phase1_qa_pdf(
                global_qa_table=global_qa_tbl if not global_qa_tbl.empty else None,
                overlay_items=overlay_items,
                macro_items=macro_items,
                report_title="Sensor Consistency & Quality Assurance Report",
            )
            st.session_state["phase1_qa_pdf"] = pdf_bytes
            st.success("Phase 1 QA report generated. Click Download below.")
        except Exception as e:
            st.error(f"Phase 1 QA report generation failed: {e}")

    if "phase1_qa_pdf" in st.session_state:
        st.download_button(
            label="Download Phase 1 QA PDF",
            data=st.session_state["phase1_qa_pdf"],
            file_name="phase1_qa_report.pdf",
            mime="application/pdf",
            key="phase1_qa_pdf_download",
        )
