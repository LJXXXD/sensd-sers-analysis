"""
Regression V2: two-stage (classify ST vs SE, then serotype-specific regression).
"""

import logging

import pandas as pd
import streamlit as st

from cache import (
    build_cached_concentration_regression_dataset,
    build_cached_two_stage_regression_artifacts,
)
from components.shared_ui import (
    render_dataframe_stretch,
    render_figure_stretch,
    render_pdf_download_section,
)
from sensd_sers_analysis.application import build_two_stage_regression_pdf_bytes
from sensd_sers_analysis.classification.plots import plot_confusion_matrix
from sensd_sers_analysis.config import REGRESSION_INLIER_FEATURE, REGRESSION_QA_FEATURES
from sensd_sers_analysis.regression.plots import plot_actual_vs_predicted, plot_residuals
from tabs.regression_common import (
    format_regression_target_counts,
    list_regression_feature_columns,
    regression_prerequisites_ok,
)

logger = logging.getLogger(__name__)


def render(filtered_features, peak_artifacts):
    _ = peak_artifacts
    st.markdown(
        "#### Regression V2: Two-stage conditional\n"
        "**Stage 1:** multi-class **serotype** classifier on positive-CFU rows. **Stage 2:** separate "
        "**Random Forest** regressors per serotype, fit on training sensors only. "
        "Test predictions **route** through the predicted class (error propagation). "
        "**Oracle** routing uses the true serotype to isolate stage-2 quality."
    )

    if not regression_prerequisites_ok(filtered_features):
        st.warning(
            "Concentration regression requires **sensor_id**, **serotype**, "
            "**concentration_group**, **log_concentration**, **PC1**, and **PC2**."
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
        st.warning("No clean positive-CFU rows for regression.")
        return

    counts_txt = format_regression_target_counts(reg_clean)
    st.caption(
        f"Clean regression data: **{len(reg_clean)}** samples"
        + (f" — {counts_txt}" if counts_txt else "")
    )

    st.markdown("---")
    try:
        artifacts = build_cached_two_stage_regression_artifacts(
            reg_clean,
            tuple(feat_cols),
        )
    except ValueError as e:
        logger.warning("Two-stage regression failed: %s", e)
        st.error(f"Two-stage regression error: {e}")
        return

    out = artifacts.outputs
    metrics_df = pd.DataFrame(
        {
            "Pipeline": ["Routed (predicted serotype)", "Oracle (true serotype)"],
            "RMSE (log10)": [out.rmse_routed, out.rmse_oracle],
            "MAE (log10)": [out.mae_routed, out.mae_oracle],
            "R²": [out.r2_routed, out.r2_oracle],
        }
    )
    st.markdown("**Held-out sensor regression metrics**")
    render_dataframe_stretch(
        metrics_df,
        column_config={
            "RMSE (log10)": st.column_config.NumberColumn(format="%.4f"),
            "MAE (log10)": st.column_config.NumberColumn(format="%.4f"),
            "R²": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        f"Stage-1 **routing accuracy** on test (best of RF/SVM by F1): **{out.routing_accuracy:.3f}**."
    )

    st.markdown("##### Stage 1: Serotype classifiers (multi-class)")
    col_rf, col_svm = st.columns(2)
    with col_rf:
        st.markdown(f"**{out.stage1_rf.model_name}**")
        fig_rf = plot_confusion_matrix(out.stage1_rf)
        render_figure_stretch(fig_rf)
    with col_svm:
        st.markdown(f"**{out.stage1_svm.model_name}**")
        fig_svm = plot_confusion_matrix(out.stage1_svm)
        render_figure_stretch(fig_svm)

    st.markdown("##### Stage 2: Regression (routed vs oracle)")
    hue = out.serotype_true_test
    fig_r = plot_actual_vs_predicted(
        out.y_true_reg,
        out.y_pred_routed,
        title="Routed pipeline (held-out sensors)",
        hue=hue,
    )
    render_figure_stretch(fig_r)
    fig_res = plot_residuals(out.y_true_reg, out.y_pred_routed, title="Residuals — routed")
    render_figure_stretch(fig_res)

    with st.expander("Oracle routing (true serotype)", expanded=False):
        fig_o = plot_actual_vs_predicted(
            out.y_true_reg,
            out.y_pred_oracle,
            title="Oracle routing (upper bound on stage 2)",
            hue=hue,
        )
        render_figure_stretch(fig_o)

    st.markdown("---")
    st.markdown("#### PDF Report")

    def _pdf():
        return build_two_stage_regression_pdf_bytes(artifacts)

    render_pdf_download_section(
        session_key="reg_two_stage_pdf",
        filename="regression_two_stage_report.pdf",
        generate_callback=_pdf,
        button_label="Generate Two-Stage Regression Report",
        download_label="Download Two-Stage Regression Report",
    )
