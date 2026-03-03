"""
Serotype Classification tab — PCA, ML classification, PDF report.
"""

import logging

import streamlit as st

from components.shared_ui import (
    render_figure_stretch,
    render_metrics_row,
    render_pdf_download_section,
)

from sensd_sers_analysis.assessment import get_global_model_consistency_qa
from sensd_sers_analysis.classification import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca_classification,
    prepare_phase2_data,
    train_classifiers,
)
from sensd_sers_analysis.processing import (
    PHASE2_FEATURE_BASE,
    get_peak_height_columns,
)
from sensd_sers_analysis.report import build_phase2_classification_pdf

logger = logging.getLogger(__name__)


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
    phase2_peak_cols = get_peak_height_columns(
        next(
            iter(st.session_state.get("peak_infos_by_serotype", {}).values()),
            [],
        )
    )
    phase2_feat_cols = [
        c
        for c in PHASE2_FEATURE_BASE + phase2_peak_cols
        if c in filtered_features.columns
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
        render_figure_stretch(fig_pca)
    except (ValueError, KeyError) as e:
        logger.warning("PCA plot error: %s", e)
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

        render_metrics_row(
            [
                ("Accuracy", f"{best.accuracy:.3f}"),
                ("Precision (weighted)", f"{best.precision:.3f}"),
                ("Recall (weighted)", f"{best.recall:.3f}"),
                ("F1-Score (weighted)", f"{best.f1:.3f}"),
            ]
        )

        st.markdown(f"**Best model:** {best.model_name} (F1={best.f1:.3f})")

        col_cm, col_fi = st.columns(2)
        with col_cm:
            st.markdown("**Confusion Matrix**")
            fig_cm = plot_confusion_matrix(best)
            render_figure_stretch(fig_cm)

        with col_fi:
            if rf_result.feature_importances is not None:
                st.markdown("**Feature Importance (Random Forest)**")
                fig_fi = plot_feature_importance(rf_result)
                render_figure_stretch(fig_fi)
            else:
                st.info("Feature importance only for Random Forest.")

        st.markdown("---")
        st.markdown("#### PDF Report")

        def _generate_phase2_pdf_bytes() -> bytes:
            fig_pca_pdf = plot_pca_classification(phase2_clean)
            fig_fi_pdf = (
                plot_feature_importance(rf_result)
                if rf_result.feature_importances is not None
                else None
            )
            fig_cm_pdf = plot_confusion_matrix(best)
            return build_phase2_classification_pdf(
                pca_fig=fig_pca_pdf,
                feature_importance_fig=fig_fi_pdf,
                confusion_matrix_fig=fig_cm_pdf,
                accuracy=best.accuracy,
                f1=best.f1,
            )

        render_pdf_download_section(
            session_key="phase2_pdf",
            filename="serotype_classification_report.pdf",
            generate_callback=_generate_phase2_pdf_bytes,
            button_label="Generate Serotype Classification Report",
            download_label="Download Serotype Classification Report",
        )
    except ValueError as e:
        logger.warning("Classification error: %s", e)
        st.error(f"Classification error: {e}")
