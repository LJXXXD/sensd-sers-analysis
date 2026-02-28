"""
Model-based sensor consistency: linear regression across concentration spectrum.

Evaluates sensor stability by fitting a regression model (log concentration vs
feature) and using the residuals (RMSE) as the core consistency metric.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sensd_sers_analysis.processing import BASIC_FEATURE_COLUMNS


@dataclass
class ConcentrationRegressionResult:
    """Result of fitting linear regression: log_concentration vs feature."""

    slope: float
    intercept: float
    r2: float
    rmse: float
    n_samples: int
    x_fit: np.ndarray  # log_concentration values used in fit
    y_pred: np.ndarray  # predicted feature values for x_fit


def fit_concentration_regression(
    df: pd.DataFrame,
    feature_col: str,
    *,
    log_conc_col: str = "log_concentration",
) -> Optional[ConcentrationRegressionResult]:
    """
    Fit linear regression of feature vs log concentration.

    Rows with missing log_concentration (e.g., 0 CFU) are excluded from the fit.

    Args:
        df: Feature DataFrame with log_concentration and feature columns.
        feature_col: Column name of the feature to predict (Y-axis).
        log_conc_col: Column name for log concentration (X-axis).

    Returns:
        ConcentrationRegressionResult or None if insufficient data (n < 2).
    """
    if feature_col not in df.columns or log_conc_col not in df.columns:
        return None

    valid = df[[log_conc_col, feature_col]].notna().all(axis=1)
    df_fit = df.loc[valid].copy()

    if len(df_fit) < 2:
        return None

    x = df_fit[log_conc_col].astype(float).values
    y = df_fit[feature_col].astype(float).values

    res = stats.linregress(x, y)
    y_pred = res.intercept + res.slope * x
    residuals = y - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = res.rvalue**2

    return ConcentrationRegressionResult(
        slope=res.slope,
        intercept=res.intercept,
        r2=r2,
        rmse=rmse,
        n_samples=len(df_fit),
        x_fit=x,
        y_pred=y_pred,
    )


def get_zero_cfu_baseline(
    df: pd.DataFrame,
    feature_col: str,
    *,
    concentration_group_col: str = "concentration_group",
    concentration_col: str = "concentration",
) -> Optional[float]:
    """
    Compute mean feature value for 0 CFU samples (baseline).

    Identifies 0 CFU rows via concentration_group == "0 CFU" or
    concentration <= 0 when concentration_group is unavailable.

    Args:
        df: Feature DataFrame.
        feature_col: Column name of the feature.
        concentration_group_col: Column with binned labels (e.g., "0 CFU").
        concentration_col: Raw concentration column for fallback.

    Returns:
        Mean feature value for 0 CFU replicates, or None if no 0 CFU samples.
    """
    if feature_col not in df.columns:
        return None

    if concentration_group_col in df.columns:
        zero_mask = df[concentration_group_col].astype(str) == "0 CFU"
    elif concentration_col in df.columns:
        conc = pd.to_numeric(df[concentration_col], errors="coerce")
        zero_mask = conc.notna() & (conc <= 0)
    else:
        return None

    zero_df = df.loc[zero_mask, feature_col].dropna()
    if zero_df.empty:
        return None

    return float(zero_df.mean())


def get_global_model_consistency(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[List[str]] = None,
    sensor_col: str = "sensor_id",
    serotype_col: str = "serotype",
    log_conc_col: str = "log_concentration",
) -> pd.DataFrame:
    """
    Compute regression metrics for every sensor_id × serotype × feature combination.

    Iterates through unique (sensor_id, serotype) pairs and all core features.
    For each combination, fits log-linear regression (excluding 0 CFU) and returns
    n_points, R_squared, RMSE. Use the resulting table to sort by RMSE and identify
    sensors with the worst consistency.

    Args:
        df: Feature DataFrame with sensor_id, serotype, log_concentration, and
            feature columns.
        feature_cols: Feature columns to assess. Defaults to BASIC_FEATURE_COLUMNS
            that exist in df.
        sensor_col: Column name for sensor identifier.
        serotype_col: Column name for serotype/serovar.
        log_conc_col: Column name for log concentration (X-axis).

    Returns:
        DataFrame with columns: sensor_id, serotype, feature, n_points,
        R_squared, RMSE. Rows with insufficient data (n < 2) are omitted.
    """
    required = [sensor_col, serotype_col, log_conc_col]
    if any(c not in df.columns for c in required):
        return pd.DataFrame(
            columns=[
                sensor_col,
                serotype_col,
                "feature",
                "n_points",
                "R_squared",
                "RMSE",
            ]
        )

    if feature_cols is None:
        feature_cols = [c for c in BASIC_FEATURE_COLUMNS if c in df.columns]
    else:
        feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        return pd.DataFrame(
            columns=[
                sensor_col,
                serotype_col,
                "feature",
                "n_points",
                "R_squared",
                "RMSE",
            ]
        )

    rows: list[dict] = []
    sensors = df[sensor_col].dropna().unique()
    serotypes = df[serotype_col].dropna().unique()

    for sens in sensors:
        for sero in serotypes:
            subset = df[
                (df[sensor_col].astype(str) == str(sens))
                & (df[serotype_col].astype(str) == str(sero))
            ]
            if subset.empty:
                continue
            for feat in feature_cols:
                res = fit_concentration_regression(
                    subset, feat, log_conc_col=log_conc_col
                )
                if res is not None:
                    rows.append(
                        {
                            sensor_col: sens,
                            serotype_col: sero,
                            "feature": feat,
                            "n_points": res.n_samples,
                            "R_squared": res.r2,
                            "RMSE": res.rmse,
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                sensor_col,
                serotype_col,
                "feature",
                "n_points",
                "R_squared",
                "RMSE",
            ]
        )
    return pd.DataFrame(rows)
