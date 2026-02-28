"""
Consistency metrics for single-sensor SERS assessment.

Computes Coefficient of Variation (CV = σ/μ) and related stats for
extracted features across replicates. Supports raw vs. outlier-filtered metrics.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sensd_sers_analysis.assessment.outliers import filter_outliers
from sensd_sers_analysis.processing import BASIC_FEATURE_COLUMNS


@dataclass
class ConsistencyResult:
    """Aggregated consistency metrics for a feature."""

    feature: str
    n_total: int
    n_inliers: int
    n_outliers: int
    mean_raw: float
    std_raw: float
    cv_raw: float  # σ/μ as fraction; use *100 for percent
    mean_filtered: float
    std_filtered: float
    cv_filtered: float
    outlier_method: str


def coefficient_of_variation(values: np.ndarray | pd.Series) -> float:
    """
    Compute CV = σ/μ as a fraction (0–1). Returns NaN if mean is 0 or invalid.

    Args:
        values: 1D numeric array.

    Returns:
        CV as fraction. Multiply by 100 for percent.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    mu = np.mean(vals)
    if mu == 0:
        return np.nan
    return float(np.std(vals) / abs(mu))


def compute_consistency_metrics(
    df: pd.DataFrame,
    feature_col: str,
    *,
    group_cols: Optional[list[str]] = None,
    outlier_method: str = "iqr",
    iqr_whis: float = 1.5,
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Compute consistency metrics (CV, mean, std) with and without outliers.

    When group_cols is None, treats the entire DataFrame as one group.
    When group_cols is provided, computes metrics per group (e.g., per
    sensor_id + concentration_group).

    Args:
        df: Feature DataFrame (from extract_basic_features).
        feature_col: Name of the feature column.
        group_cols: Columns to group by (e.g., ["sensor_id", "concentration_group"]).
        outlier_method: "iqr" or "zscore".
        iqr_whis: IQR multiplier for IQR method.
        zscore_threshold: Z-score cutoff for zscore method.

    Returns:
        DataFrame with columns: [group_cols (if any), feature, n_total,
        n_outliers, mean_raw, std_raw, cv_raw, mean_filtered, std_filtered,
        cv_filtered, outlier_method].
    """
    if feature_col not in df.columns:
        raise ValueError(
            f"feature_col '{feature_col}' not in DataFrame. "
            f"Available: {list(df.columns)}"
        )

    def _row_metrics(g: pd.DataFrame) -> pd.Series:
        inliers, outliers = filter_outliers(
            g,
            feature_col,
            method=outlier_method,
            iqr_whis=iqr_whis,
            zscore_threshold=zscore_threshold,
        )
        vals = g[feature_col].dropna()
        vals_in = inliers[feature_col].dropna()

        mu_raw = vals.mean() if len(vals) > 0 else np.nan
        std_raw = vals.std() if len(vals) > 0 else np.nan
        cv_raw = coefficient_of_variation(vals) if len(vals) > 0 else np.nan

        mu_f = vals_in.mean() if len(vals_in) > 0 else np.nan
        std_f = vals_in.std() if len(vals_in) > 0 else np.nan
        cv_f = coefficient_of_variation(vals_in) if len(vals_in) > 0 else np.nan

        return pd.Series(
            {
                "feature": feature_col,
                "n_total": len(g),
                "n_inliers": len(inliers),
                "n_outliers": len(outliers),
                "mean_raw": mu_raw,
                "std_raw": std_raw,
                "cv_raw": cv_raw,
                "mean_filtered": mu_f,
                "std_filtered": std_f,
                "cv_filtered": cv_f,
                "outlier_method": outlier_method,
            }
        )

    if group_cols is None:
        return pd.DataFrame([_row_metrics(df)])

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"group_cols {missing} not in DataFrame. Available: {list(df.columns)}"
        )

    return df.groupby(group_cols, dropna=False).apply(_row_metrics).reset_index()


def get_consistency_summary_table(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    *,
    group_cols: Optional[list[str]] = None,
    outlier_method: str = "iqr",
) -> pd.DataFrame:
    """
    Build a summary table of consistency metrics for multiple features.

    Args:
        df: Feature DataFrame.
        feature_cols: Features to include. Default: BASIC_FEATURE_COLUMNS
            present in df.
        group_cols: Grouping columns.
        outlier_method: Outlier detection method.

    Returns:
        Concatenated DataFrame of metrics for all features.
    """
    if feature_cols is None:
        feature_cols = [c for c in BASIC_FEATURE_COLUMNS if c in df.columns]
    if not feature_cols:
        return pd.DataFrame()

    parts = []
    for fc in feature_cols:
        if fc not in df.columns:
            continue
        part = compute_consistency_metrics(
            df, fc, group_cols=group_cols, outlier_method=outlier_method
        )
        parts.append(part)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
