"""
Application orchestration for SENS-D validation metric tables.
"""

from __future__ import annotations

import pandas as pd

from sensd_sers_analysis.application.regression_service import (
    build_concentration_regression_dataset,
)
from sensd_sers_analysis.application.classification_service import (
    build_classification_clean_dataset,
)
from sensd_sers_analysis.assessment.validation_metrics import (
    ValidationTableArtifacts,
    build_validation_tables,
)
from sensd_sers_analysis.config import (
    CLASSIFICATION_INLIER_FEATURE,
    CLASSIFICATION_QA_FEATURES,
    REGRESSION_INLIER_FEATURE,
    REGRESSION_QA_FEATURES,
    VALIDATION_ACCURACY_MIN_THRESHOLD,
)


def build_validation_table_artifacts(
    filtered_features: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    repeatability_feature: str = CLASSIFICATION_INLIER_FEATURE,
    accuracy_threshold: float = VALIDATION_ACCURACY_MIN_THRESHOLD,
) -> ValidationTableArtifacts:
    """
    Build Metrics-docx Tables 1–3 from filtered feature data.

    Parameters
    ----------
    filtered_features:
        Feature dataframe after app filters (with targeted peaks merged).
    feature_columns:
        ML feature columns for identification and quantification models.
    repeatability_feature:
        Scalar feature for CV and signal-change metrics.
    accuracy_threshold:
        Minimum mean identification accuracy for Meet Target? (Table 2).

    Returns
    -------
    ValidationTableArtifacts
        Tables 1–3; empty dataframes when prerequisites are not met.
    """
    classification_clean = build_classification_clean_dataset(
        filtered_features,
        excluded_map_policy=CLASSIFICATION_QA_FEATURES,
        inlier_feature=CLASSIFICATION_INLIER_FEATURE,
    )
    regression_clean = build_concentration_regression_dataset(
        filtered_features,
        excluded_map_policy=REGRESSION_QA_FEATURES,
        inlier_feature=REGRESSION_INLIER_FEATURE,
    )
    return build_validation_tables(
        classification_clean,
        regression_clean,
        feature_cols=list(feature_columns),
        repeatability_feature=repeatability_feature,
        accuracy_threshold=accuracy_threshold,
    )
