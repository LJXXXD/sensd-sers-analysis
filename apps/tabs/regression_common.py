"""
Shared prerequisites and feature lists for concentration regression tabs.
"""

import pandas as pd

from sensd_sers_analysis.processing import (
    PHASE2_FEATURE_BASE,
    list_targeted_peak_feature_columns,
)


def regression_prerequisites_ok(filtered_features) -> bool:
    """Return True when the dataframe has Phase 2-style columns for regression."""
    return (
        "sensor_id" in filtered_features.columns
        and "serotype" in filtered_features.columns
        and "concentration_group" in filtered_features.columns
        and "log_concentration" in filtered_features.columns
        and "PC1" in filtered_features.columns
        and "PC2" in filtered_features.columns
    )


def list_regression_feature_columns(filtered_features_columns) -> list[str]:
    """Same feature union as Phase 2 classification (integral, PCs, peak_near_*)."""
    peak_cols = list_targeted_peak_feature_columns(filtered_features_columns)
    return [c for c in PHASE2_FEATURE_BASE + peak_cols if c in filtered_features_columns]


def format_regression_target_counts(reg_clean: pd.DataFrame, target_col: str = "target") -> str:
    """Comma-separated class counts for the regression ``target`` column (sorted keys)."""
    if reg_clean.empty or target_col not in reg_clean.columns:
        return ""
    vc = reg_clean[target_col].astype(str).value_counts().sort_index()
    return ", ".join(f"{k}: {int(v)}" for k, v in vc.items())
