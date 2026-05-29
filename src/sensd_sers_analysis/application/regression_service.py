"""
Application orchestration for concentration regression paradigms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sensd_sers_analysis.application.contracts import (
    GlobalRegressionArtifacts,
    MtlRegressionArtifacts,
    TwoStageRegressionArtifacts,
)
from sensd_sers_analysis.assessment import get_global_model_consistency_qa
from sensd_sers_analysis.config import (
    REGRESSION_INLIER_FEATURE,
    REGRESSION_QA_FEATURES,
    REGRESSION_RANDOM_STATE,
    REGRESSION_TEST_SIZE,
)
from sensd_sers_analysis.regression import (
    assert_disjoint_group_split,
    group_train_test_indices,
    prepare_concentration_regression_data,
    train_global_regressors,
    train_mtl_regressor,
    train_two_stage_regressors,
)


def build_concentration_regression_dataset(
    filtered_features: pd.DataFrame,
    *,
    excluded_map_policy: tuple[str, ...] = REGRESSION_QA_FEATURES,
    inlier_feature: str = REGRESSION_INLIER_FEATURE,
) -> pd.DataFrame:
    """
    Build the clean concentration-regression table (positive CFU, dynamic serotypes).

    Parameters
    ----------
    filtered_features:
        Feature dataframe after app filters.
    excluded_map_policy:
        Features used to build the global QA exclusion map.
    inlier_feature:
        Feature column used for intra-sensor outlier filtering (shared policy with
        serotype classification).

    Returns
    -------
    pd.DataFrame
        Regression-ready rows; empty when prerequisites fail.
    """

    _, excluded_map = get_global_model_consistency_qa(
        filtered_features,
        feature_cols=list(excluded_map_policy),
    )
    return prepare_concentration_regression_data(
        filtered_features,
        excluded_map=excluded_map,
        feature_cols=list(excluded_map_policy),
        inlier_feature=inlier_feature,
    )


def _split_train_test(
    regression_clean: pd.DataFrame,
    *,
    test_size: float = REGRESSION_TEST_SIZE,
    random_state: int = REGRESSION_RANDOM_STATE,
    group_col: str = "sensor_id",
) -> tuple[np.ndarray, np.ndarray]:
    """Shared group holdout for all three paradigms."""
    groups = regression_clean[group_col].astype(str).to_numpy(dtype=object, copy=False)
    train_idx, test_idx = group_train_test_indices(
        groups,
        test_size=test_size,
        random_state=random_state,
    )
    assert_disjoint_group_split(regression_clean, train_idx, test_idx, group_col=group_col)
    return train_idx, test_idx


def run_global_concentration_regression(
    regression_clean: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> GlobalRegressionArtifacts:
    """Paradigm 1: serotype-blind global RF + SVR."""
    df = regression_clean.reset_index(drop=True)
    train_idx, test_idx = _split_train_test(df)
    rf_result, svm_result = train_global_regressors(
        df,
        list(feature_columns),
        train_idx,
        test_idx,
        random_state=REGRESSION_RANDOM_STATE,
    )
    best = rf_result if rf_result.rmse <= svm_result.rmse else svm_result
    return GlobalRegressionArtifacts(
        regression_clean=df,
        feature_columns=feature_columns,
        train_indices=np.asarray(train_idx, dtype=np.intp),
        test_indices=np.asarray(test_idx, dtype=np.intp),
        rf_result=rf_result,
        svm_result=svm_result,
        best_result=best,
    )


def run_two_stage_concentration_regression(
    regression_clean: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> TwoStageRegressionArtifacts:
    """Paradigm 2: classify serotype, then route to serotype-specific regressors."""
    df = regression_clean.reset_index(drop=True)
    train_idx, test_idx = _split_train_test(df)
    outputs = train_two_stage_regressors(
        df,
        list(feature_columns),
        train_idx,
        test_idx,
        random_state=REGRESSION_RANDOM_STATE,
    )
    return TwoStageRegressionArtifacts(
        regression_clean=df,
        feature_columns=feature_columns,
        train_indices=np.asarray(train_idx, dtype=np.intp),
        test_indices=np.asarray(test_idx, dtype=np.intp),
        outputs=outputs,
    )


def run_mtl_concentration_regression(
    regression_clean: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> MtlRegressionArtifacts:
    """Paradigm 3: PyTorch MTL shared trunk."""
    df = regression_clean.reset_index(drop=True)
    train_idx, test_idx = _split_train_test(df)
    outputs = train_mtl_regressor(
        df,
        list(feature_columns),
        train_idx,
        test_idx,
        random_state=REGRESSION_RANDOM_STATE,
    )
    return MtlRegressionArtifacts(
        regression_clean=df,
        feature_columns=feature_columns,
        train_indices=np.asarray(train_idx, dtype=np.intp),
        test_indices=np.asarray(test_idx, dtype=np.intp),
        outputs=outputs,
    )


def build_global_regression_pdf_bytes(artifacts: GlobalRegressionArtifacts) -> bytes:
    """PDF with metrics table + global regression plots."""
    from sensd_sers_analysis.regression.plots import plot_actual_vs_predicted, plot_residuals
    from sensd_sers_analysis.report import build_regression_concentration_pdf

    best = artifacts.best_result
    compare_df = pd.DataFrame(
        {
            "Model": [artifacts.rf_result.model_name, artifacts.svm_result.model_name],
            "RMSE (log10)": [artifacts.rf_result.rmse, artifacts.svm_result.rmse],
            "MAE (log10)": [artifacts.rf_result.mae, artifacts.svm_result.mae],
            "R²": [artifacts.rf_result.r2, artifacts.svm_result.r2],
        }
    )
    fig_scatter = plot_actual_vs_predicted(
        best.y_true,
        best.y_pred,
        title=f"Global model — {best.model_name} (held-out sensors)",
    )
    fig_res = plot_residuals(best.y_true, best.y_pred)
    pdf = build_regression_concentration_pdf(
        paradigm_title="Paradigm 1: Global (serotype-blind) regression",
        metrics_table=compare_df,
        scatter_fig=fig_scatter,
        residual_fig=fig_res,
    )
    import matplotlib.pyplot as plt

    plt.close(fig_scatter)
    plt.close(fig_res)
    return pdf


def build_two_stage_regression_pdf_bytes(artifacts: TwoStageRegressionArtifacts) -> bytes:
    from sensd_sers_analysis.classification.plots import plot_confusion_matrix
    from sensd_sers_analysis.regression.plots import plot_actual_vs_predicted, plot_residuals
    from sensd_sers_analysis.report import build_regression_concentration_pdf

    out = artifacts.outputs
    metrics_df = pd.DataFrame(
        {
            "Pipeline": ["Routed (predicted serotype)", "Oracle (true serotype)"],
            "RMSE (log10)": [out.rmse_routed, out.rmse_oracle],
            "MAE (log10)": [out.mae_routed, out.mae_oracle],
            "R²": [out.r2_routed, out.r2_oracle],
        }
    )
    fig_scatter = plot_actual_vs_predicted(
        out.y_true_reg,
        out.y_pred_routed,
        title="Two-stage — routed regression (held-out sensors)",
    )
    fig_res = plot_residuals(out.y_true_reg, out.y_pred_routed, title="Residuals — routed")
    fig_cm = plot_confusion_matrix(out.stage1_best)
    pdf = build_regression_concentration_pdf(
        paradigm_title="Paradigm 2: Two-stage (classify then regress)",
        metrics_table=metrics_df,
        scatter_fig=fig_scatter,
        residual_fig=fig_res,
        extra_figures=[("Stage 1 confusion (best classifier)", fig_cm)],
        caption_lines=(
            f"Stage-1 routing accuracy on test: {out.routing_accuracy:.3f}. "
            "Oracle uses true serotype for routing (upper bound on stage 2).",
        ),
    )
    import matplotlib.pyplot as plt

    plt.close(fig_scatter)
    plt.close(fig_res)
    plt.close(fig_cm)
    return pdf


def build_mtl_regression_pdf_bytes(artifacts: MtlRegressionArtifacts) -> bytes:
    import matplotlib.pyplot as plt

    from sensd_sers_analysis.regression.plots import plot_actual_vs_predicted, plot_residuals
    from sensd_sers_analysis.report import build_regression_concentration_pdf

    out = artifacts.outputs
    labels = out.class_labels
    sero = np.asarray([labels[int(i)] for i in out.y_true_cls], dtype=object)
    n_cls = len(labels)
    metrics_df = pd.DataFrame(
        {
            "Task": ["Regression (log10)", f"Classification ({n_cls} serotypes)"],
            "Metric": ["RMSE / MAE / R²", "Accuracy"],
            "Value": [
                f"{out.rmse:.4f} / {out.mae:.4f} / {out.r2:.4f}",
                f"{out.clf_accuracy:.4f}",
            ],
        }
    )
    fig_scatter = plot_actual_vs_predicted(
        out.y_true_reg,
        out.y_pred_reg,
        title="MTL — regression head (held-out sensors)",
        hue=sero,
    )
    fig_res = plot_residuals(out.y_true_reg, out.y_pred_reg, title="Residuals — MTL regression")
    pdf = build_regression_concentration_pdf(
        paradigm_title="Paradigm 3: Multi-task learning (shared trunk)",
        metrics_table=metrics_df,
        scatter_fig=fig_scatter,
        residual_fig=fig_res,
    )
    plt.close(fig_scatter)
    plt.close(fig_res)
    return pdf
