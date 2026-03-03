"""
Phase 2: Serotyping & Classification tab — PCA, ML classification, PDF report.
"""

import streamlit as st

from sensd_sers_analysis.assessment import get_global_model_consistency_qa
from sensd_sers_analysis.classification import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca_classification,
    prepare_phase2_data,
    train_classifiers,
)
from sensd_sers_analysis.processing import get_peak_height_columns
from sensd_sers_analysis.report import build_phase2_classification_pdf


def render(filtered_features):
    """Render the Phase 2 Serotyping & Classification tab."""
    st.markdown(
        "#### Phase 2: Serotyping & Classification\n"
        "Uses strictly clean data from Phase 1: Pass sensors only, inlier "
        "points only. Trains baseline ML models for 3-class classification: "
        "**ST**, **SE**, **Rinsate** (0 CFU)."
    )

    has_phase2_cols = (
        "sensor_id" in filtered_features.columns
        and "serotype" in filtered_features.columns
        and "concentration_group" in filtered_features.columns
        and "PC1" in filtered_features.columns
        and "PC2" in filtered_features.columns
    )
    phase2_feat_base = [
        "integral_area",
        "max_intensity",
        "mean_intensity",
        "PC1",
        "PC2",
    ]
    phase2_peak_cols = get_peak_height_columns(
        next(
            iter(st.session_state.get("peak_infos_by_serotype", {}).values()),
            [],
        )
    )
    phase2_feat_cols = [
        c for c in phase2_feat_base + phase2_peak_cols if c in filtered_features.columns
    ]

    if not has_phase2_cols:
        st.warning(
            "Phase 2 requires **sensor_id**, **serotype**, **concentration_group**, "
            "**PC1**, and **PC2**. Run Model-Based Sensor Consistency first "
            "(which computes Pass/Excluded) and ensure data has PCA features."
        )
        return
    if len(phase2_feat_cols) < 2:
        st.warning(
            "Need at least 2 feature columns (integral_area, PC1, etc.) for "
            "classification. Check that features are extracted."
        )
        return

    _, excluded_map_p2 = get_global_model_consistency_qa(
        filtered_features,
        feature_cols=["integral_area"],
    )
    phase2_clean = prepare_phase2_data(
        filtered_features,
        excluded_map=excluded_map_p2,
        inlier_feature="integral_area",
    )

    if phase2_clean.empty:
        st.warning(
            "No clean data for Phase 2. Ensure Phase 1 has Pass sensors and "
            "inlier points. Check that ST, SE, and Rinsate (0 CFU) samples exist."
        )
        return

    counts = phase2_clean["target"].value_counts()
    st.caption(
        f"Clean data: **{len(phase2_clean)}** samples — "
        + ", ".join(f"{k}: {v}" for k, v in counts.items())
    )

    st.markdown("---")
    st.markdown("##### 1. Unsupervised Clustering (PCA Scatter)")
    try:
        fig_pca = plot_pca_classification(phase2_clean)
        st.pyplot(fig_pca, width="stretch")
    except (ValueError, KeyError) as e:
        st.error(f"PCA plot error: {e}")

    st.markdown("---")
    st.markdown("##### 2. Baseline ML Classification")
    st.caption(
        "80/20 stratified split. Features: integral_area, max_intensity, "
        "mean_intensity, PC1, PC2, plus dynamic peak heights."
    )

    try:
        rf_result, svm_result = train_classifiers(
            phase2_clean, phase2_feat_cols, target_col="target"
        )
        best = rf_result if rf_result.f1 >= svm_result.f1 else svm_result

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Accuracy", f"{best.accuracy:.3f}")
        with m2:
            st.metric("Precision (weighted)", f"{best.precision:.3f}")
        with m3:
            st.metric("Recall (weighted)", f"{best.recall:.3f}")
        with m4:
            st.metric("F1-Score (weighted)", f"{best.f1:.3f}")

        st.markdown(f"**Best model:** {best.model_name} (F1={best.f1:.3f})")

        col_cm, col_fi = st.columns(2)
        with col_cm:
            st.markdown("**Confusion Matrix**")
            fig_cm = plot_confusion_matrix(best)
            st.pyplot(fig_cm, width="stretch")

        with col_fi:
            if rf_result.feature_importances is not None:
                st.markdown("**Feature Importance (Random Forest)**")
                fig_fi = plot_feature_importance(rf_result)
                st.pyplot(fig_fi, width="stretch")
            else:
                st.info("Feature importance only for Random Forest.")

        st.markdown("---")
        st.markdown("##### Phase 2 Classification Report")
        if st.button("Generate Phase 2 Classification Report", key="phase2_pdf_btn"):
            try:
                fig_pca_pdf = plot_pca_classification(phase2_clean)
                fig_fi_pdf = (
                    plot_feature_importance(rf_result)
                    if rf_result.feature_importances is not None
                    else None
                )
                fig_cm_pdf = plot_confusion_matrix(best)
                pdf_bytes = build_phase2_classification_pdf(
                    pca_fig=fig_pca_pdf,
                    feature_importance_fig=fig_fi_pdf,
                    confusion_matrix_fig=fig_cm_pdf,
                    accuracy=best.accuracy,
                    f1=best.f1,
                )
                st.session_state["phase2_pdf"] = pdf_bytes
                st.success("Phase 2 report generated. Click Download below.")
            except Exception as e:
                st.error(f"Phase 2 report generation failed: {e}")

        if "phase2_pdf" in st.session_state:
            st.download_button(
                label="Download Phase 2 Classification Report",
                data=st.session_state["phase2_pdf"],
                file_name="phase2_classification_report.pdf",
                mime="application/pdf",
                key="phase2_pdf_download",
            )
    except ValueError as e:
        st.error(f"Classification error: {e}")
