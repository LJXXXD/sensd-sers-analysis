"""
Serotype Classification tab — PCA, ML classification, PDF report.
"""

import logging

import streamlit as st

from cache import build_cached_phase2_artifacts, build_cached_phase2_dataset
from components.shared_ui import (
    render_figure_stretch,
    render_metrics_row,
    render_pdf_download_section,
)

from sensd_sers_analysis.application import build_phase2_pdf_bytes
from sensd_sers_analysis.classification import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca_classification,
)
from sensd_sers_analysis.config import PHASE2_INLIER_FEATURE, PHASE2_QA_FEATURES
from sensd_sers_analysis.processing import (
    PHASE2_FEATURE_BASE,
    list_targeted_peak_feature_columns,
)

logger = logging.getLogger(__name__)


def render(filtered_features, peak_artifacts):
    """
    Render the Phase 2 Serotyping & Classification tab.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe for the current app state.
    peak_artifacts:
        Reserved for a uniform tab signature (classification uses ``peak_near_*``
        columns already merged into ``filtered_features``).
    """

    _ = peak_artifacts

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
    phase2_peak_cols = list_targeted_peak_feature_columns(filtered_features.columns)
    phase2_feat_cols = [
        c for c in PHASE2_FEATURE_BASE + phase2_peak_cols if c in filtered_features.columns
    ]

    if not has_phase2_cols:
        st.warning(
            "Phase 2 requires **sensor_id**, **serotype**, **concentration_group**, "
            "**PC1**, and **PC2**. Run **Sensor assessment** first "
            "(which computes Pass/Excluded) and ensure data has PCA features."
        )
        return
    if len(phase2_feat_cols) < 2:
        st.warning(
            "Need at least 2 feature columns (integral_area, PC1, etc.) for "
            "classification. Check that features are extracted."
        )
        return

    phase2_clean = build_cached_phase2_dataset(
        filtered_features,
        excluded_map_policy=PHASE2_QA_FEATURES,
        inlier_feature=PHASE2_INLIER_FEATURE,
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
        "mean_intensity, PC1, PC2, plus targeted ``peak_near_*`` heights."
    )

    try:
        phase2_artifacts = build_cached_phase2_artifacts(
            phase2_clean,
            tuple(phase2_feat_cols),
        )

        render_metrics_row(
            [
                ("Accuracy", f"{phase2_artifacts.best_result.accuracy:.3f}"),
                (
                    "Precision (weighted)",
                    f"{phase2_artifacts.best_result.precision:.3f}",
                ),
                ("Recall (weighted)", f"{phase2_artifacts.best_result.recall:.3f}"),
                ("F1-Score (weighted)", f"{phase2_artifacts.best_result.f1:.3f}"),
            ]
        )

        st.markdown(
            "**Best model:** "
            f"{phase2_artifacts.best_result.model_name} "
            f"(F1={phase2_artifacts.best_result.f1:.3f})"
        )

        col_cm, col_fi = st.columns(2)
        with col_cm:
            st.markdown("**Confusion Matrix**")
            fig_cm = plot_confusion_matrix(phase2_artifacts.best_result)
            render_figure_stretch(fig_cm)

        with col_fi:
            if phase2_artifacts.rf_result.feature_importances is not None:
                st.markdown("**Feature Importance (Random Forest)**")
                fig_fi = plot_feature_importance(phase2_artifacts.rf_result)
                render_figure_stretch(fig_fi)
            else:
                st.info("Feature importance only for Random Forest.")

        st.markdown("---")
        st.markdown("#### PDF Report")

        def _generate_phase2_pdf_bytes() -> bytes:
            return build_phase2_pdf_bytes(phase2_artifacts)

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
