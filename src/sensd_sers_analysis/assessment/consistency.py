"""
Consistency metrics for single-sensor SERS assessment.

Computes Coefficient of Variation (CV = σ/μ) and related stats for
extracted features across replicates. Supports raw vs. outlier-filtered metrics.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from sensd_sers_analysis.assessment.outliers import filter_outliers
from sensd_sers_analysis.processing import BASIC_FEATURE_COLUMNS, extract_scalar_concentration

# Default columns for grouping in sensor assessment consistency tables.
# Grouping uses the nominal target concentration so sample-to-sample CFU spread
# (measured from the actual concentration) is captured within each group rather
# than splitting replicates across actual-concentration bins.
ASSESSMENT_GROUP_COLS: List[str] = [
    "sensor_id",
    "serotype",
    "target_concentration_group",
]

# Column holding the measured (plate-count) CFU/mL per row.
ACTUAL_CONCENTRATION_COL = "concentration"


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
    # Data-side variability of the measured concentration within the group.
    conc_mean: float = np.nan
    conc_std: float = np.nan
    conc_cv_raw: float = np.nan  # σ/μ of actual CFU/mL as fraction
    conc_cv_filtered: float = np.nan


def coefficient_of_variation(values: np.ndarray | pd.Series) -> float:
    """
    Compute CV = σ/μ as a fraction (0–1).

    Returns NaN when variability is undefined rather than a misleading value:

    - Fewer than 2 finite values: a single measurement has no spread, so a
      naive population std would be 0 and imply perfect repeatability. This is
      not real information (the sensor was measured once), so NaN is returned.
    - Mean of 0: CV is undefined (division by zero).

    Args:
        values: 1D numeric array.

    Returns:
        CV as a fraction (multiply by 100 for percent), or NaN when undefined.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
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
    concentration_col: str = ACTUAL_CONCENTRATION_COL,
) -> pd.DataFrame:
    """
    Compute consistency metrics (CV, mean, std) with and without outliers.

    In addition to the SERS-signal CV of ``feature_col``, this computes the CV
    of the **actual** (measured) concentration within each group. Comparing the
    two isolates the sensor-added variability: the signal CV reflects sensor
    noise plus true sample-to-sample CFU spread, while the concentration CV
    reflects the sample spread alone.

    When group_cols is None, treats the entire DataFrame as one group.
    When group_cols is provided, computes metrics per group (e.g., per
    sensor_id + target_concentration_group).

    Args:
        df: Feature DataFrame (from extract_basic_features).
        feature_col: Name of the SERS-signal feature column.
        group_cols: Columns to group by (e.g., ["sensor_id",
            "target_concentration_group"]).
        outlier_method: "iqr" or "zscore".
        iqr_whis: IQR multiplier for IQR method.
        zscore_threshold: Z-score cutoff for zscore method.
        concentration_col: Column with the measured CFU/mL used for the
            data-side concentration CV.

    Returns:
        DataFrame with columns: [group_cols (if any), feature, n_total,
        n_inliers, n_outliers, mean_raw, std_raw, cv_raw, mean_filtered,
        std_filtered, cv_filtered, conc_mean, conc_std, conc_cv_raw,
        conc_cv_filtered, outlier_method].
    """
    if feature_col not in df.columns:
        raise ValueError(
            f"feature_col '{feature_col}' not in DataFrame. Available: {list(df.columns)}"
        )

    has_conc = concentration_col in df.columns

    def _concentration_cv(
        g: pd.DataFrame,
        inlier_index: pd.Index,
    ) -> tuple[float, float, float, float]:
        """Return (conc_mean, conc_std, conc_cv_raw, conc_cv_filtered) for a group."""
        if not has_conc:
            return np.nan, np.nan, np.nan, np.nan
        conc = extract_scalar_concentration(g[concentration_col], g).dropna()
        if conc.empty:
            return np.nan, np.nan, np.nan, np.nan
        conc_mean = float(conc.mean())
        conc_std = float(conc.std()) if len(conc) > 1 else np.nan
        conc_cv_raw = coefficient_of_variation(conc)
        conc_in = conc.loc[conc.index.intersection(inlier_index)]
        conc_cv_filtered = coefficient_of_variation(conc_in) if len(conc_in) > 0 else np.nan
        return conc_mean, conc_std, conc_cv_raw, conc_cv_filtered

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

        conc_mean, conc_std, conc_cv_raw, conc_cv_filtered = _concentration_cv(g, inliers.index)

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
                "conc_mean": conc_mean,
                "conc_std": conc_std,
                "conc_cv_raw": conc_cv_raw,
                "conc_cv_filtered": conc_cv_filtered,
                "outlier_method": outlier_method,
            }
        )

    if group_cols is None:
        return pd.DataFrame([_row_metrics(df)])

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"group_cols {missing} not in DataFrame. Available: {list(df.columns)}")

    return (
        df.groupby(group_cols, dropna=False, observed=True)
        .apply(_row_metrics, include_groups=False)
        .reset_index()
    )


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
