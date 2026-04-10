"""
Serotype Classification tab — PCA, ML classification, PDF report.
"""

import logging

import pandas as pd
import streamlit as st

from cache import build_cached_phase2_artifacts, build_cached_phase2_dataset
from components.shared_ui import (
    render_dataframe_stretch,
    render_figure_stretch,
    render_pdf_download_section,
)

from sensd_sers_analysis.application import build_phase2_pdf_bytes
from sensd_sers_analysis.classification import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca_classification,
)
from sensd_sers_analysis.config import (
    PHASE2_HYPERPARAMETER_TUNING,
    PHASE2_INLIER_FEATURE,
    PHASE2_QA_FEATURES,
    PHASE2_TUNING_MIN_TRAIN_SAMPLES,
)
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
        "points only. Trains baseline ML models for **(N + 1)-class** "
        "classification: **N** serotypes on positive-CFU rows plus **Rinsate** (0 CFU), "
        "where **N** is determined by the current filter."
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
            "inlier points. Check that positive-CFU serotypes and Rinsate (0 CFU) samples exist."
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
        "80/20 stratified train/test split; features are standardized before fitting. "
        "Feature set: integral_area, max_intensity, mean_intensity, PC1, PC2, plus "
        "targeted ``peak_near_*`` heights. **Both** Random Forest and SVM (RBF) are "
        "trained on the same split; tables and plots report metrics on the held-out "
        "test set only."
    )

    try:
        phase2_artifacts = build_cached_phase2_artifacts(
            phase2_clean,
            tuple(phase2_feat_cols),
        )

        rf = phase2_artifacts.rf_result
        svm = phase2_artifacts.svm_result
        compare_df = pd.DataFrame(
            {
                "Model": [rf.model_name, svm.model_name],
                "Accuracy": [rf.accuracy, svm.accuracy],
                "Precision (weighted)": [rf.precision, svm.precision],
                "Recall (weighted)": [rf.recall, svm.recall],
                "F1 (weighted)": [rf.f1, svm.f1],
            }
        )
        st.markdown("**Held-out test metrics**")
        render_dataframe_stretch(
            compare_df,
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.3f"),
                "Precision (weighted)": st.column_config.NumberColumn(format="%.3f"),
                "Recall (weighted)": st.column_config.NumberColumn(format="%.3f"),
                "F1 (weighted)": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption(
            f"**Best by weighted F1:** {phase2_artifacts.best_result.model_name} "
            f"(F1 = {phase2_artifacts.best_result.f1:.3f})."
        )

        with st.expander("Hyperparameter tuning (how models are fit)", expanded=False):
            if PHASE2_HYPERPARAMETER_TUNING:
                st.markdown(
                    f"When the **training** split has at least "
                    f"{PHASE2_TUNING_MIN_TRAIN_SAMPLES} samples, each model is tuned with "
                    "**RandomizedSearchCV** and stratified cross-validation on that training "
                    "split only. Search grids and iteration counts live in "
                    "``config/model_policies.py``. Smaller training sets skip search and use "
                    "fixed defaults (``PHASE2_RF_N_ESTIMATORS`` for RF; default RBF-SVM "
                    "settings from scikit-learn)."
                )
            else:
                st.markdown(
                    "Hyperparameter search is **disabled** "
                    "(``PHASE2_HYPERPARAMETER_TUNING`` in ``config/model_policies.py``). "
                    "Models use fixed defaults only."
                )
            st.markdown("**Selected hyperparameters (after tuning, if any)**")
            st.markdown(
                f"- **{rf.model_name}:** "
                + (
                    "`"
                    + ", ".join(
                        f"{k}={v!r}"
                        for k, v in sorted(rf.best_params.items(), key=lambda kv: kv[0])
                    )
                    + "`"
                    if rf.best_params
                    else "*none — defaults used (search skipped or disabled).*"
                )
            )
            st.markdown(
                f"- **{svm.model_name}:** "
                + (
                    "`"
                    + ", ".join(
                        f"{k}={v!r}"
                        for k, v in sorted(svm.best_params.items(), key=lambda kv: kv[0])
                    )
                    + "`"
                    if svm.best_params
                    else "*none — defaults used (search skipped or disabled).*"
                )
            )

        col_rf, col_svm = st.columns(2)
        with col_rf:
            st.markdown(f"**Confusion matrix — {rf.model_name}**")
            fig_rf_cm = plot_confusion_matrix(rf)
            render_figure_stretch(fig_rf_cm)
        with col_svm:
            st.markdown(f"**Confusion matrix — {svm.model_name}**")
            fig_svm_cm = plot_confusion_matrix(svm)
            render_figure_stretch(fig_svm_cm)

        if phase2_artifacts.rf_result.feature_importances is not None:
            st.markdown("**Feature importance (Random Forest)**")
            fig_fi = plot_feature_importance(phase2_artifacts.rf_result)
            render_figure_stretch(fig_fi)
        else:
            st.info("Feature importance is only available for Random Forest.")

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
