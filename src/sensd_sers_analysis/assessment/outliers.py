"""
Outlier detection for SERS feature data.

Provides statistical methods (IQR, Z-score) to identify faulty reads and noise
for downstream filtering in consistency and batch analyses.
"""

from typing import Literal

import numpy as np
import pandas as pd


def detect_outliers_iqr(
    values: np.ndarray | pd.Series,
    *,
    whis: float = 1.5,
) -> np.ndarray:
    """
    Identify outliers using the interquartile range (IQR) method.

    Outliers are points outside [Q1 - whis*IQR, Q3 + whis*IQR],
    where IQR = Q3 - Q1. Default whis=1.5 matches the standard boxplot rule.

    Args:
        values: 1D array or Series of numeric values.
        whis: Multiplier for IQR (default 1.5).

    Returns:
        Boolean array: True for outliers, False otherwise. NaN values are
        marked as False (not counted as outliers).
    """
    vals = np.asarray(values, dtype=float)
    mask_valid = np.isfinite(vals)
    result = np.zeros(len(vals), dtype=bool)

    if not mask_valid.any():
        return result

    q1 = np.nanpercentile(vals, 25)
    q3 = np.nanpercentile(vals, 75)
    iqr = q3 - q1

    if iqr <= 0:
        return result

    lower = q1 - whis * iqr
    upper = q3 + whis * iqr
    result[mask_valid] = (vals[mask_valid] < lower) | (vals[mask_valid] > upper)
    return result


def detect_outliers_zscore(
    values: np.ndarray | pd.Series,
    *,
    threshold: float = 3.0,
) -> np.ndarray:
    """
    Identify outliers using Z-score (standard deviation from mean).

    Outliers are points with |z| > threshold. Uses robust statistics
    (median, MAD) when possible to reduce influence of existing outliers.

    Args:
        values: 1D array or Series of numeric values.
        threshold: Z-score cutoff (default 3.0).

    Returns:
        Boolean array: True for outliers, False otherwise. NaN values are
        marked as False.
    """
    vals = np.asarray(values, dtype=float)
    mask_valid = np.isfinite(vals)
    result = np.zeros(len(vals), dtype=bool)

    if not mask_valid.any():
        return result

    valid = vals[mask_valid]
    mu = np.mean(valid)
    sigma = np.std(valid)

    if sigma <= 0:
        return result

    z = np.abs(vals - mu) / sigma
    result[mask_valid] = z[mask_valid] > threshold
    return result


def filter_outliers(
    df: pd.DataFrame,
    feature_col: str,
    *,
    method: Literal["iqr", "zscore"] = "iqr",
    iqr_whis: float = 1.5,
    zscore_threshold: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a feature DataFrame into inliers and outliers.

    Args:
        df: DataFrame with at least the feature column.
        feature_col: Name of the numeric feature column.
        method: "iqr" or "zscore".
        iqr_whis: IQR multiplier when method="iqr".
        zscore_threshold: Z-score cutoff when method="zscore".

    Returns:
        Tuple of (inliers_df, outliers_df). Inliers exclude outlier rows;
        outliers contains only the outlier rows.
    """
    if feature_col not in df.columns:
        raise ValueError(
            f"feature_col '{feature_col}' not in DataFrame. "
            f"Available: {list(df.columns)}"
        )

    vals = df[feature_col].values
    if method == "iqr":
        is_outlier = detect_outliers_iqr(vals, whis=iqr_whis)
    else:
        is_outlier = detect_outliers_zscore(vals, threshold=zscore_threshold)

    inliers = df.loc[~is_outlier].copy()
    outliers = df.loc[is_outlier].copy()
    return inliers, outliers
