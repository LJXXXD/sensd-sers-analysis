"""
Application-layer orchestration for serotype classification.
"""

from __future__ import annotations

from sensd_sers_analysis.utils.availability import PlotUnavailableError
from contextlib import ExitStack
import pandas as pd
from sensd_sers_analysis.report.figures import own_figure

from sensd_sers_analysis.application.contracts import ClassificationArtifacts
from sensd_sers_analysis.classification import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca_classification,
    train_classifiers,
)
from sensd_sers_analysis.config import CLASSIFICATION_INLIER_FEATURE, CLASSIFICATION_QA_FEATURES
from sensd_sers_analysis.classification.data_prep import label_classification_dataset
from sensd_sers_analysis.modeling import select_by_training_cv
from sensd_sers_analysis.report import build_classification_report_pdf


def build_classification_clean_dataset(
    filtered_features: pd.DataFrame,
    *,
    excluded_map_policy: tuple[str, ...] = CLASSIFICATION_QA_FEATURES,
    inlier_feature: str = CLASSIFICATION_INLIER_FEATURE,
) -> pd.DataFrame:
    """Build identity-eligible model rows without retrospective response screening.

    QA arguments remain accepted for existing callers but do not determine model
    eligibility. Use ``prepare_classification_dataset`` for retrospective diagnostics.
    """
    return label_classification_dataset(filtered_features)


def run_classification_training(
    clean_classification_df: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> ClassificationArtifacts:
    """
    Train serotype classifiers and choose the best result.

    Parameters
    ----------
    clean_classification_df:
        Identity-eligible dataframe for classification.
    feature_columns:
        Feature columns used during training.

    Returns
    -------
    ClassificationArtifacts
        Clean data, both model results, and the selected best result.
    """

    rf_result, svm_result = train_classifiers(
        clean_classification_df,
        list(feature_columns),
        target_col="target",
    )
    best_result = select_by_training_cv(rf_result, svm_result)
    return ClassificationArtifacts(
        clean_classification_df=clean_classification_df,
        feature_columns=feature_columns,
        rf_result=rf_result,
        svm_result=svm_result,
        best_result=best_result,
    )


def build_classification_report_pdf_bytes(artifacts: ClassificationArtifacts) -> bytes:
    """
    Build the serotype classification PDF from cached artifacts.

    Parameters
    ----------
    artifacts:
        Cached classification outputs.

    Returns
    -------
    bytes
        PDF document bytes.
    """
    with ExitStack() as figures:
        pca_reason = None
        try:
            pca_fig = plot_pca_classification(artifacts.clean_classification_df)
        except PlotUnavailableError as exc:
            pca_fig = None
            pca_reason = f"Unavailable: {exc}"
        if pca_fig is not None:
            own_figure(figures, pca_fig)
        feature_importance_fig = None
        if artifacts.rf_result.feature_importances is not None:
            feature_importance_fig = plot_feature_importance(artifacts.rf_result)
            own_figure(figures, feature_importance_fig)
        rf_cm_fig = plot_confusion_matrix(artifacts.rf_result)
        own_figure(figures, rf_cm_fig)
        svm_cm_fig = plot_confusion_matrix(artifacts.svm_result)
        own_figure(figures, svm_cm_fig)
        return build_classification_report_pdf(
            pca_fig=pca_fig,
            feature_importance_fig=feature_importance_fig,
            rf_confusion_matrix_fig=rf_cm_fig,
            svm_confusion_matrix_fig=svm_cm_fig,
            rf_accuracy=artifacts.rf_result.accuracy,
            rf_f1=artifacts.rf_result.f1,
            svm_accuracy=artifacts.svm_result.accuracy,
            svm_f1=artifacts.svm_result.f1,
            best_model_name=artifacts.best_result.model_name,
            pca_unavailable_reason=pca_reason,
            caption_lines=(
                f"Outer split seed={artifacts.rf_result.split_seed}; train rows={len(artifacts.rf_result.train_indices)}, test rows={len(artifacts.rf_result.test_indices)}.",
                "Predictors: " + ", ".join(artifacts.feature_columns) + ".",
                f"Training CV weighted F1: RF={artifacts.rf_result.cv_score}, SVM={artifacts.svm_result.cv_score}. Unavailable CV uses the fixed RF reference.",
            ),
        )
