"""
Regression V3: multi-task learning (shared trunk + classification + regression).
"""

import logging

import numpy as np
import pandas as pd
import streamlit as st

from cache import (
    build_cached_concentration_regression_dataset,
    build_cached_mtl_regression_artifacts,
)
from components.shared_ui import (
    render_dataframe_stretch,
    render_figure_stretch,
    render_pdf_download_section,
)
from sensd_sers_analysis.application import build_mtl_regression_pdf_bytes
from sensd_sers_analysis.config import (
    REGRESSION_INLIER_FEATURE,
    REGRESSION_MTL_LAMBDA_CLASSIFICATION,
    REGRESSION_MTL_LAMBDA_REGRESSION,
    REGRESSION_QA_FEATURES,
)
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
        "#### Regression V3: Multi-task learning (MTL)\n"
        "A **shared PyTorch MLP trunk** with two heads: **multi-class serotype** classification "
        "and **log10 concentration** regression. Joint loss "
        f"``λ_cls·CE + λ_reg·MSE`` with λ_cls={REGRESSION_MTL_LAMBDA_CLASSIFICATION}, "
        f"λ_reg={REGRESSION_MTL_LAMBDA_REGRESSION} (see ``config/model_policies.py``). "
        "Same **sensor group holdout** as V1/V2."
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
        artifacts = build_cached_mtl_regression_artifacts(
            reg_clean,
            tuple(feat_cols),
        )
    except ValueError as e:
        logger.warning("MTL regression failed: %s", e)
        st.error(f"MTL regression error: {e}")
        return

    out = artifacts.outputs
    labels = out.class_labels
    sero = np.asarray([labels[int(i)] for i in out.y_true_cls], dtype=object)
    summary = pd.DataFrame(
        {
            "Task": [
                "Regression RMSE (log10)",
                "Regression MAE (log10)",
                "Regression R²",
                f"Classification accuracy ({len(labels)} serotypes)",
            ],
            "Test value": [out.rmse, out.mae, out.r2, out.clf_accuracy],
        }
    )
    st.markdown("**Held-out sensor metrics**")
    render_dataframe_stretch(
        summary,
        column_config={"Test value": st.column_config.NumberColumn(format="%.4f")},
    )

    st.markdown("##### Regression head")
    fig_s = plot_actual_vs_predicted(
        out.y_true_reg,
        out.y_pred_reg,
        title="MTL regression head (held-out sensors)",
        hue=sero,
    )
    render_figure_stretch(fig_s)
    fig_res = plot_residuals(out.y_true_reg, out.y_pred_reg, title="Residuals — MTL")
    render_figure_stretch(fig_res)

    st.markdown("---")
    st.markdown("#### PDF Report")

    def _pdf():
        return build_mtl_regression_pdf_bytes(artifacts)

    render_pdf_download_section(
        session_key="reg_mtl_pdf",
        filename="regression_mtl_report.pdf",
        generate_callback=_pdf,
        button_label="Generate MTL Regression Report",
        download_label="Download MTL Regression Report",
    )
