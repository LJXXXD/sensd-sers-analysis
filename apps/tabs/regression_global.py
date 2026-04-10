"""
Regression V1: global (serotype-blind) concentration prediction.
"""

import logging

import pandas as pd
import streamlit as st

from cache import (
    build_cached_concentration_regression_dataset,
    build_cached_global_regression_artifacts,
)
from components.shared_ui import (
    render_dataframe_stretch,
    render_figure_stretch,
    render_pdf_download_section,
)
from sensd_sers_analysis.application import build_global_regression_pdf_bytes
from sensd_sers_analysis.config import (
    REGRESSION_HYPERPARAMETER_TUNING,
    REGRESSION_INLIER_FEATURE,
    REGRESSION_QA_FEATURES,
    REGRESSION_TUNING_GROUP_KFOLD_SPLITS,
    REGRESSION_TUNING_MIN_TRAIN_SAMPLES,
)
from sensd_sers_analysis.regression.plots import (
    plot_actual_vs_predicted,
    plot_regression_feature_importance,
    plot_residuals,
)
from tabs.regression_common import (
    format_regression_target_counts,
    list_regression_feature_columns,
    regression_prerequisites_ok,
)

logger = logging.getLogger(__name__)


def render(filtered_features, peak_artifacts):
    _ = peak_artifacts
    st.markdown(
        "#### Regression V1: Global (serotype-blind)\n"
        "A **single** regressor pools **all positive-CFU** serotypes in the filtered data "
        "and predicts **log10 concentration** without serotype information in the "
        "feature vector. Train/test split is a **group holdout by sensor_id** "
        "(test sensors are unseen during training)."
    )

    if not regression_prerequisites_ok(filtered_features):
        st.warning(
            "Concentration regression requires **sensor_id**, **serotype**, "
            "**concentration_group**, **log_concentration**, **PC1**, and **PC2**. "
            "Run **Sensor assessment** and ensure metadata is complete."
        )
        return

    feat_cols = list_regression_feature_columns(filtered_features.columns)
    if len(feat_cols) < 2:
        st.warning("Need at least two numeric feature columns for regression.")
        return

    reg_clean = build_cached_concentration_regression_dataset(
        filtered_features,
        excluded_map_policy=REGRESSION_QA_FEATURES,
        inlier_feature=REGRESSION_INLIER_FEATURE,
    )
    if reg_clean.empty:
        st.warning(
            "No clean positive-CFU rows for regression. Check Phase 1 Pass "
            "sensors, inliers, and non-zero concentrations."
        )
        return

    counts_txt = format_regression_target_counts(reg_clean)
    st.caption(
        f"Clean regression data: **{len(reg_clean)}** samples"
        + (f" — {counts_txt}" if counts_txt else "")
    )

    st.markdown("---")
    st.markdown("##### Global models (RF + SVR)")
    st.caption(
        "Features are standardized on the **training sensors only**. "
        "Hyperparameter search (when enabled) uses **GroupKFold** on training rows "
        f"({REGRESSION_TUNING_GROUP_KFOLD_SPLITS} folds max) to avoid sensor leakage "
        "within tuning."
    )

    try:
        artifacts = build_cached_global_regression_artifacts(
            reg_clean,
            tuple(feat_cols),
        )
    except ValueError as e:
        logger.warning("Global regression failed: %s", e)
        st.error(f"Global regression error: {e}")
        return

    rf = artifacts.rf_result
    svm = artifacts.svm_result
    compare_df = pd.DataFrame(
        {
            "Model": [rf.model_name, svm.model_name],
            "RMSE (log10)": [rf.rmse, svm.rmse],
            "MAE (log10)": [rf.mae, svm.mae],
            "R²": [rf.r2, svm.r2],
        }
    )
    st.markdown("**Held-out sensor metrics (test set)**")
    render_dataframe_stretch(
        compare_df,
        column_config={
            "RMSE (log10)": st.column_config.NumberColumn(format="%.4f"),
            "MAE (log10)": st.column_config.NumberColumn(format="%.4f"),
            "R²": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    best = artifacts.best_result
    st.caption(f"**Best by test RMSE:** {best.model_name} (RMSE = {best.rmse:.4f}).")

    with st.expander("Hyperparameter tuning (global regression)", expanded=False):
        if REGRESSION_HYPERPARAMETER_TUNING:
            st.markdown(
                f"When the **training** split has at least "
                f"{REGRESSION_TUNING_MIN_TRAIN_SAMPLES} samples and at least two "
                "training **sensor** groups, both regressors are tuned with "
                "**RandomizedSearchCV** and **GroupKFold** on the training split only. "
                "Grids live in ``config/model_policies.py`` under ``REGRESSION_*``."
            )
        else:
            st.markdown(
                "Hyperparameter search is **disabled** "
                "(``REGRESSION_HYPERPARAMETER_TUNING`` in ``config/model_policies.py``)."
            )
        for res in (rf, svm):
            st.markdown(
                f"- **{res.model_name}:** "
                + (
                    "`" + ", ".join(f"{k}={v!r}" for k, v in sorted(res.best_params.items())) + "`"
                    if res.best_params
                    else "*defaults (search skipped or disabled).*"
                )
            )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Actual vs predicted — {rf.model_name}**")
        fig_a = plot_actual_vs_predicted(
            rf.y_true,
            rf.y_pred,
            title=f"{rf.model_name} (held-out sensors)",
            hue=artifacts.regression_clean.iloc[artifacts.test_indices]["target"].to_numpy(),
        )
        render_figure_stretch(fig_a)
    with col_b:
        st.markdown(f"**Actual vs predicted — {svm.model_name}**")
        fig_b = plot_actual_vs_predicted(
            svm.y_true,
            svm.y_pred,
            title=f"{svm.model_name} (held-out sensors)",
            hue=artifacts.regression_clean.iloc[artifacts.test_indices]["target"].to_numpy(),
        )
        render_figure_stretch(fig_b)

    st.markdown("**Residuals (best model)**")
    fig_res = plot_residuals(best.y_true, best.y_pred)
    render_figure_stretch(fig_res)

    st.markdown("**Feature importance (Random Forest)**")
    try:
        fig_fi = plot_regression_feature_importance(rf)
        render_figure_stretch(fig_fi)
    except ValueError as e:
        st.info(str(e))

    st.markdown("---")
    st.markdown("#### PDF Report")

    def _pdf():
        return build_global_regression_pdf_bytes(artifacts)

    render_pdf_download_section(
        session_key="reg_global_pdf",
        filename="regression_global_report.pdf",
        generate_callback=_pdf,
        button_label="Generate Global Regression Report",
        download_label="Download Global Regression Report",
    )
