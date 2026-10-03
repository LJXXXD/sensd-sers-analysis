"""
Shared prerequisites and feature lists for concentration regression tabs.
"""

import pandas as pd

from sensd_sers_analysis.processing import (
    CLASSIFICATION_FEATURE_BASE,
    list_targeted_peak_feature_columns,
)


def regression_prerequisites_ok(filtered_features) -> bool:
    """Return True when the dataframe has the shared columns needed for regression."""
    return (
        "sensor_id" in filtered_features.columns
        and "serotype" in filtered_features.columns
        and "sample_type" in filtered_features.columns
        and "concentration" in filtered_features.columns
        and "log_concentration" in filtered_features.columns
    )


def list_regression_feature_columns(filtered_features_columns) -> list[str]:
    """Independent per-spectrum predictors shared with classification."""
    peak_cols = list_targeted_peak_feature_columns(filtered_features_columns)
    return CLASSIFICATION_FEATURE_BASE + peak_cols


def format_regression_target_counts(reg_clean: pd.DataFrame, target_col: str = "target") -> str:
    """Comma-separated class counts for the regression ``target`` column (sorted keys)."""
    if reg_clean.empty or target_col not in reg_clean.columns:
        return ""
    vc = reg_clean[target_col].astype(str).value_counts().sort_index()
    return ", ".join(f"{k}: {int(v)}" for k, v in vc.items())
