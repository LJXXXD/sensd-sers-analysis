"""
Degradation analysis for SERS sensors over repeated use.

Fits a linear trend (feature vs. sequence/timestamp) to detect performance
degradation. A negative slope suggests degradation. The temporal axis must
be test_id or date—not signal_index (spatial within a file).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def prepare_degradation_data(
    df: pd.DataFrame,
    feature_col: str,
    *,
    sensor_col: str = "sensor_id",
    test_col: str = "test_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Aggregate feature to one value per (sensor_id, test_id) and add test_ordinal.

    Caller must pre-filter to a single serotype and concentration_group.
    test_ordinal is 1, 2, 3, ... per sensor, ordered by date (or test_id if no date).
    If test_id is missing, groups by (sensor_id, date) instead.

    Args:
        df: Feature DataFrame, already filtered to one serotype + concentration.
        feature_col: Feature to aggregate (mean).
        sensor_col: Sensor identifier.
        test_col: Test/session identifier (use date_col grouping if missing).
        date_col: Date column for ordering or fallback grouping.

    Returns:
        DataFrame with one row per (sensor_id, test_id): sensor_id, test_id,
        <feature_col> (mean), test_ordinal.
    """
    required = [sensor_col, feature_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"prepare_degradation_data: missing columns {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Use test_id if present, else date for temporal grouping
    if test_col in df.columns:
        group_cols = [sensor_col, test_col]
    elif date_col in df.columns:
        group_cols = [sensor_col, date_col]
        test_col = date_col  # treat date as "test" for ordinal
    else:
        raise ValueError(
            "prepare_degradation_data: need test_id or date for temporal axis. "
            f"Available: {list(df.columns)}"
        )

    agg_dict = {feature_col: "mean"}
    if date_col in df.columns and date_col not in group_cols:
        agg_dict[date_col] = "first"

    df_agg = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

    # Assign test_ordinal: 1, 2, 3, ... per sensor, ordered by date (or test_id)
    order_col = date_col if date_col in df_agg.columns else test_col
    df_agg = df_agg.sort_values([sensor_col, order_col], na_position="last")
    df_agg["test_ordinal"] = df_agg.groupby(sensor_col, sort=False).cumcount() + 1
    return df_agg


@dataclass
class DegradationResult:
    """Result of linear regression for degradation analysis."""

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    slope_stderr: float
    n_points: int
    slope_interpretation: str  # "degradation", "improvement", "stable"


def compute_degradation(
    df: pd.DataFrame,
    feature_col: str,
    sequence_col: str,
    *,
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Fit feature ~ sequence and compute slope for degradation assessment.

    Negative slope indicates degradation (feature decreases over sequence).
    Positive slope indicates improvement. Near-zero slope indicates stability.

    Args:
        df: Feature DataFrame with feature_col and sequence_col.
        feature_col: Y-axis (e.g., max_intensity, integral_area).
        sequence_col: X-axis (e.g., signal_index, or date ordinal).
        group_cols: Columns to group by (e.g., ["sensor_id"]). When provided,
            fits separately per group.

    Returns:
        DataFrame with columns: [group_cols (if any), feature, sequence_col,
        slope, intercept, r_squared, p_value, slope_stderr, n_points,
        slope_interpretation].
    """
    if feature_col not in df.columns:
        raise ValueError(
            f"feature_col '{feature_col}' not in DataFrame. "
            f"Available: {list(df.columns)}"
        )
    if sequence_col not in df.columns:
        raise ValueError(
            f"sequence_col '{sequence_col}' not in DataFrame. "
            f"Available: {list(df.columns)}"
        )

    def _fit_group(g: pd.DataFrame) -> pd.Series:
        g_clean = g.dropna(subset=[feature_col, sequence_col])
        x = g_clean[sequence_col].astype(float).values
        y = g_clean[feature_col].astype(float).values

        if len(x) < 2 or len(y) < 2:
            return pd.Series(
                {
                    "feature": feature_col,
                    "sequence_col": sequence_col,
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r_squared": np.nan,
                    "p_value": np.nan,
                    "slope_stderr": np.nan,
                    "n_points": len(g_clean),
                    "slope_interpretation": "insufficient_data",
                }
            )

        res = stats.linregress(x, y)
        slope = res.slope
        r2 = res.rvalue**2 if res.rvalue is not None else np.nan

        # Interpret slope: use 5% relative threshold for "stable"
        y_mean = np.mean(y)
        if y_mean != 0 and np.isfinite(slope):
            rel_slope = abs(slope) / abs(y_mean) * 100
            if rel_slope < 0.5:
                interp = "stable"
            elif slope < 0:
                interp = "degradation"
            else:
                interp = "improvement"
        else:
            interp = (
                "stable"
                if abs(slope) < 1e-10
                else ("degradation" if slope < 0 else "improvement")
            )

        return pd.Series(
            {
                "feature": feature_col,
                "sequence_col": sequence_col,
                "slope": slope,
                "intercept": res.intercept,
                "r_squared": r2,
                "p_value": res.pvalue,
                "slope_stderr": res.stderr,
                "n_points": len(g_clean),
                "slope_interpretation": interp,
            }
        )

    if group_cols is None:
        return pd.DataFrame([_fit_group(df)])

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"group_cols {missing} not in DataFrame. Available: {list(df.columns)}"
        )

    return df.groupby(group_cols, dropna=False).apply(_fit_group).reset_index()


def add_sequence_column(
    df: pd.DataFrame,
    *,
    sequence_col: str = "signal_index",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Ensure a numeric sequence column exists for degradation X-axis.

    Uses sequence_col if present and numeric. Otherwise, converts date_col
    to ordinal (days since min date). If neither works, uses row index.

    Args:
        df: DataFrame with metadata.
        sequence_col: Preferred column for sequence (e.g., signal_index).
        date_col: Fallback for date-based ordering.

    Returns:
        Copy of df with "sequence" column added (0-based integer sequence).
    """
    out = df.copy()

    if sequence_col in df.columns:
        out["sequence"] = pd.to_numeric(df[sequence_col], errors="coerce")
        if out["sequence"].notna().any():
            # Sort and assign 0,1,2,... per group if needed
            out["sequence"] = out["sequence"].fillna(0).astype(int)
            return out

    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if dates.notna().any():
            out["sequence"] = (dates - dates.min()).dt.total_seconds() / 86400
            return out

    out["sequence"] = np.arange(len(df))
    return out
