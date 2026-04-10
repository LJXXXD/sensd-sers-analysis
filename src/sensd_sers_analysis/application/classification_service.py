"""
Application-layer orchestration for the Phase 2 classification workflow.
"""

from __future__ import annotations

import pandas as pd

from sensd_sers_analysis.application.contracts import Phase2Artifacts
from sensd_sers_analysis.assessment import get_global_model_consistency_qa
from sensd_sers_analysis.classification import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca_classification,
    prepare_phase2_data,
    train_classifiers,
)
from sensd_sers_analysis.config import PHASE2_INLIER_FEATURE, PHASE2_QA_FEATURES
from sensd_sers_analysis.report import build_phase2_classification_pdf


def build_phase2_dataset(
    filtered_features: pd.DataFrame,
    *,
    excluded_map_policy: tuple[str, ...] = PHASE2_QA_FEATURES,
    inlier_feature: str = PHASE2_INLIER_FEATURE,
) -> pd.DataFrame:
    """
    Build the cleaned Phase 2 dataset from filtered feature data.

    Parameters
    ----------
    filtered_features:
        Feature dataframe after global app filters are applied.
    excluded_map_policy:
        Feature set used to build the QA exclusion map.
    inlier_feature:
        Feature used for the Phase 2 inlier filter.

    Returns
    -------
    pd.DataFrame
        Clean dataframe ready for classification.
    """

    _, excluded_map = get_global_model_consistency_qa(
        filtered_features,
        feature_cols=list(excluded_map_policy),
    )
    return prepare_phase2_data(
        filtered_features,
        excluded_map=excluded_map,
        inlier_feature=inlier_feature,
    )


def run_phase2_classification(
    phase2_clean: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> Phase2Artifacts:
    """
    Train Phase 2 classifiers and choose the best result.

    Parameters
    ----------
    phase2_clean:
        Clean Phase 2 dataframe.
    feature_columns:
        Feature columns used during training.

    Returns
    -------
    Phase2Artifacts
        Clean data, both model results, and the selected best result.
    """

    rf_result, svm_result = train_classifiers(
        phase2_clean,
        list(feature_columns),
        target_col="target",
    )
    best_result = rf_result if rf_result.f1 >= svm_result.f1 else svm_result
    return Phase2Artifacts(
        phase2_clean=phase2_clean,
        feature_columns=feature_columns,
        rf_result=rf_result,
        svm_result=svm_result,
        best_result=best_result,
    )


def build_phase2_pdf_bytes(artifacts: Phase2Artifacts) -> bytes:
    """
    Build the Phase 2 classification PDF from cached artifacts.

    Parameters
    ----------
    artifacts:
        Cached Phase 2 classification outputs.

    Returns
    -------
    bytes
        PDF document bytes.
    """

    pca_fig = plot_pca_classification(artifacts.phase2_clean)
    feature_importance_fig = None
    if artifacts.rf_result.feature_importances is not None:
        feature_importance_fig = plot_feature_importance(artifacts.rf_result)
    rf_cm_fig = plot_confusion_matrix(artifacts.rf_result)
    svm_cm_fig = plot_confusion_matrix(artifacts.svm_result)
    return build_phase2_classification_pdf(
        pca_fig=pca_fig,
        feature_importance_fig=feature_importance_fig,
        rf_confusion_matrix_fig=rf_cm_fig,
        svm_confusion_matrix_fig=svm_cm_fig,
        rf_accuracy=artifacts.rf_result.accuracy,
        rf_f1=artifacts.rf_result.f1,
        svm_accuracy=artifacts.svm_result.accuracy,
        svm_f1=artifacts.svm_result.f1,
        best_model_name=artifacts.best_result.model_name,
    )
