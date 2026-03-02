"""
Model-based sensor consistency: linear regression across concentration spectrum.

Evaluates sensor stability by fitting a regression model (log concentration vs
feature) and using the residuals (RMSE) as the core consistency metric.
Supports two-pass residual-based outlier cleaning and inter-sensor batch exclusion.
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


@dataclass
class CleanedRegressionResult:
    """
    Result of two-pass regression with residual-based outlier removal.

    Pass 1: fit all valid >0 CFU points, compute raw RMSE and raw R².
    Pass 2: refit on inliers, compute clean RMSE and R².
    """

    raw_rmse: float
    raw_r2: float
    clean_rmse: float
    clean_r2: float
    n_samples: int
    n_outliers: int
    outlier_mask: np.ndarray  # True = outlier
    raw_result: ConcentrationRegressionResult  # fit on all points (pass 1)
    clean_result: ConcentrationRegressionResult  # fit on inliers (pass 2)


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


def _detect_residual_outliers_iqr(
    residuals: np.ndarray, *, whis: float = 1.5
) -> np.ndarray:
    """
    Identify outlier points using IQR on absolute residuals.

    Outliers are points with |residual| > Q3 + whis×IQR of |residuals|.

    Args:
        residuals: 1D array of residuals.
        whis: IQR multiplier (default 1.5).

    Returns:
        Boolean array: True for outliers.
    """
    abs_res = np.abs(residuals)
    abs_res_valid = abs_res[np.isfinite(abs_res)]
    if len(abs_res_valid) < 4:
        return np.zeros(len(residuals), dtype=bool)

    q1 = np.percentile(abs_res_valid, 25)
    q3 = np.percentile(abs_res_valid, 75)
    iqr = q3 - q1
    if iqr <= 0:
        return np.zeros(len(residuals), dtype=bool)

    upper = q3 + whis * iqr
    return np.abs(residuals) > upper


def fit_concentration_regression_cleaned(
    df: pd.DataFrame,
    feature_col: str,
    *,
    log_conc_col: str = "log_concentration",
    iqr_whis: float = 1.5,
) -> Optional[CleanedRegressionResult]:
    """
    Two-pass regression with residual-based outlier removal (excluding 0 CFU).

    Pass 1: Fit all valid >0 CFU points. Compute raw RMSE. Identify outliers
    via IQR on absolute residuals (points with |residual| > Q3 + 1.5×IQR).
    Pass 2: Remove outliers, refit on inliers. Compute clean RMSE and R².

    Args:
        df: Feature DataFrame with log_concentration and feature columns.
        feature_col: Column name of the feature (Y-axis).
        log_conc_col: Column name for log concentration (X-axis).
        iqr_whis: IQR multiplier for residual outlier detection.

    Returns:
        CleanedRegressionResult or None if insufficient data (n < 2).
    """
    if feature_col not in df.columns or log_conc_col not in df.columns:
        return None

    valid = df[[log_conc_col, feature_col]].notna().all(axis=1)
    df_fit = df.loc[valid].copy()

    if len(df_fit) < 2:
        return None

    x = df_fit[log_conc_col].astype(float).values
    y = df_fit[feature_col].astype(float).values

    # Pass 1: fit all valid points
    res1 = stats.linregress(x, y)
    y_pred1 = res1.intercept + res1.slope * x
    residuals = y - y_pred1
    raw_rmse = np.sqrt(np.mean(residuals**2))

    # Outlier detection on absolute residuals
    outlier_mask = _detect_residual_outliers_iqr(residuals, whis=iqr_whis)
    n_outliers = int(np.sum(outlier_mask))

    raw_r2 = res1.rvalue**2 if res1.rvalue is not None else np.nan
    raw_result = ConcentrationRegressionResult(
        slope=res1.slope,
        intercept=res1.intercept,
        r2=raw_r2,
        rmse=raw_rmse,
        n_samples=len(df_fit),
        x_fit=x,
        y_pred=y_pred1,
    )

    # Pass 2: refit on inliers (need at least 2 points)
    inlier_mask = ~outlier_mask
    n_inliers = int(np.sum(inlier_mask))
    if n_inliers < 2:
        return CleanedRegressionResult(
            raw_rmse=raw_rmse,
            raw_r2=raw_r2,
            clean_rmse=raw_rmse,
            clean_r2=raw_r2,
            n_samples=len(df_fit),
            n_outliers=n_outliers,
            outlier_mask=outlier_mask,
            raw_result=raw_result,
            clean_result=raw_result,
        )

    x_clean = x[inlier_mask]
    y_clean = y[inlier_mask]
    res2 = stats.linregress(x_clean, y_clean)
    y_pred2 = res2.intercept + res2.slope * x_clean
    clean_rmse = np.sqrt(np.mean((y_clean - y_pred2) ** 2))
    clean_r2 = res2.rvalue**2 if res2.rvalue is not None else np.nan

    clean_result = ConcentrationRegressionResult(
        slope=res2.slope,
        intercept=res2.intercept,
        r2=clean_r2,
        rmse=clean_rmse,
        n_samples=n_inliers,
        x_fit=x_clean,
        y_pred=y_pred2,
    )
    return CleanedRegressionResult(
        raw_rmse=raw_rmse,
        raw_r2=raw_r2,
        clean_rmse=clean_rmse,
        clean_r2=clean_r2,
        n_samples=len(df_fit),
        n_outliers=n_outliers,
        outlier_mask=outlier_mask,
        raw_result=raw_result,
        clean_result=clean_result,
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


def get_global_model_consistency_qa(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[List[str]] = None,
    sensor_col: str = "sensor_id",
    serotype_col: str = "serotype",
    log_conc_col: str = "log_concentration",
    rejection_multiplier: float = 2.0,
    r2_min_threshold: float = 0.80,
    iqr_whis: float = 1.5,
) -> tuple[pd.DataFrame, dict[tuple[str, str], set[str]]]:
    """
    QA pipeline: cleaned regression per sensor + dual-threshold batch exclusion.

    For each (sensor, serotype, feature): fit with residual-based outlier
    removal. A sensor is marked "Excluded" if:
    - Clean RMSE > rejection_multiplier × batch median (too noisy), OR
    - Clean R² < r2_min_threshold (dead/unresponsive flat sensor).

    Args:
        df: Feature DataFrame with sensor_id, serotype, log_concentration.
        feature_cols: Features to assess. Default: BASIC_FEATURE_COLUMNS in df.
        sensor_col: Sensor identifier column.
        serotype_col: Serotype column.
        log_conc_col: Log concentration column (X-axis).
        rejection_multiplier: Exclude if Clean RMSE > this × batch median.
        r2_min_threshold: Exclude if Clean R² < this (flat/dead sensor).
        iqr_whis: IQR multiplier for residual outlier detection.

    Returns:
        Tuple of (DataFrame, excluded_sensors_map). DataFrame has columns:
        sensor_id, serotype, feature, n_points, outliers, raw_rmse,
        raw_r2, clean_rmse, clean_r2, status. excluded_sensors_map is
        {(serotype, feature): {sensor_id, ...}}.
    """
    _cols = [
        sensor_col,
        serotype_col,
        "feature",
        "n_points",
        "outliers",
        "raw_rmse",
        "raw_r2",
        "clean_rmse",
        "clean_r2",
        "status",
    ]
    required = [sensor_col, serotype_col, log_conc_col]
    if any(c not in df.columns for c in required):
        return (pd.DataFrame(columns=_cols), {})

    if feature_cols is None:
        feature_cols = [c for c in BASIC_FEATURE_COLUMNS if c in df.columns]
    else:
        feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        return (pd.DataFrame(columns=_cols), {})

    rows: list[dict] = []
    for sens in df[sensor_col].dropna().unique():
        for sero in df[serotype_col].dropna().unique():
            subset = df[
                (df[sensor_col].astype(str) == str(sens))
                & (df[serotype_col].astype(str) == str(sero))
            ]
            if subset.empty:
                continue
            for feat in feature_cols:
                cres = fit_concentration_regression_cleaned(
                    subset, feat, log_conc_col=log_conc_col, iqr_whis=iqr_whis
                )
                if cres is not None:
                    rows.append(
                        {
                            sensor_col: sens,
                            serotype_col: sero,
                            "feature": feat,
                            "n_points": cres.n_samples,
                            "outliers": cres.n_outliers,
                            "raw_rmse": cres.raw_rmse,
                            "raw_r2": cres.raw_r2,
                            "clean_rmse": cres.clean_rmse,
                            "clean_r2": cres.clean_r2,
                        }
                    )

    if not rows:
        return (pd.DataFrame(columns=_cols), {})

    tbl = pd.DataFrame(rows)

    # Compute median Clean RMSE per (serotype, feature)
    median_per_group = (
        tbl.groupby([serotype_col, "feature"])["clean_rmse"]
        .median()
        .reset_index()
        .rename(columns={"clean_rmse": "batch_median_rmse"})
    )
    tbl = tbl.merge(median_per_group, on=[serotype_col, "feature"], how="left")

    # Dual threshold: Excluded if (A) too noisy OR (B) dead/flat (low R²)
    rmse_exclude = tbl["clean_rmse"] > (rejection_multiplier * tbl["batch_median_rmse"])
    r2_exclude = tbl["clean_r2"] < r2_min_threshold
    tbl["status"] = np.where(
        rmse_exclude | r2_exclude,
        "Excluded",
        "Pass",
    )
    tbl = tbl.drop(columns=["batch_median_rmse"])

    # Build excluded_sensors map: {(serotype, feature): {sensor_id, ...}}
    excluded_mask = tbl["status"] == "Excluded"
    excluded_df = tbl.loc[excluded_mask, [serotype_col, "feature", sensor_col]]
    excluded_map: dict[tuple[str, str], set[str]] = {}
    for (sero, feat), group in excluded_df.groupby([serotype_col, "feature"]):
        key = (str(sero), str(feat))
        excluded_map[key] = set(group[sensor_col].astype(str).tolist())

    return tbl, excluded_map


@dataclass
class MacroRegressionResult:
    """Result of two-pass pooled macro-regression across Pass sensors' inlier data."""

    slope: float  # Clean fit slope
    intercept: float  # Clean fit intercept
    raw_slope: float
    raw_intercept: float
    raw_batch_rmse: float
    raw_batch_r2: float
    clean_batch_rmse: float
    clean_batch_r2: float
    n_points: int
    n_sensors: int
    n_macro_outliers: int
    macro_outlier_mask: np.ndarray
    x_pooled: np.ndarray
    y_pooled: np.ndarray


def compute_macro_batch_regression(
    df: pd.DataFrame,
    serotype: str,
    feature_col: str,
    pass_sensors: set[str],
    *,
    sensor_col: str = "sensor_id",
    serotype_col: str = "serotype",
    log_conc_col: str = "log_concentration",
    iqr_whis: float = 1.5,
) -> Optional[MacroRegressionResult]:
    """
    Fit a single macro-regression to pooled inlier data from Pass sensors only.

    For each Pass sensor, runs cleaned fit to identify inliers, then pools all
    inlier (x, y) points and fits one regression. Provides batch-level RMSE
    and R² for the good batch.

    Args:
        df: Feature DataFrame.
        serotype: Serotype to filter.
        feature_col: Feature column (Y-axis).
        pass_sensors: Sensor IDs that passed QA (use only their inlier data).
        sensor_col: Sensor identifier column.
        serotype_col: Serotype column.
        log_conc_col: Log concentration column (X-axis).
        iqr_whis: IQR multiplier for residual outlier detection per sensor.

    Returns:
        MacroRegressionResult or None if insufficient pooled data (n < 2).
    """
    required = [sensor_col, serotype_col, log_conc_col, feature_col]
    if any(c not in df.columns for c in required):
        return None

    subset_all = df[
        (df[serotype_col].astype(str) == str(serotype))
        & df[log_conc_col].notna()
        & df[feature_col].notna()
    ]
    if subset_all.empty:
        return None

    x_pooled: list[float] = []
    y_pooled: list[float] = []
    sensors_contributing = 0

    for sens in pass_sensors:
        sub = subset_all[subset_all[sensor_col].astype(str) == str(sens)]
        if sub.empty:
            continue
        cres = fit_concentration_regression_cleaned(
            sub, feature_col, log_conc_col=log_conc_col, iqr_whis=iqr_whis
        )
        if cres is None:
            continue
        valid = sub[[log_conc_col, feature_col]].notna().all(axis=1)
        sub_fit = sub.loc[valid]
        x_vals = sub_fit[log_conc_col].astype(float).values
        y_vals = sub_fit[feature_col].astype(float).values
        inlier_mask = ~cres.outlier_mask
        x_in = x_vals[inlier_mask]
        y_in = y_vals[inlier_mask]
        x_pooled.extend(x_in.tolist())
        y_pooled.extend(y_in.tolist())
        sensors_contributing += 1

    if len(x_pooled) < 2:
        return None

    x_arr = np.array(x_pooled)
    y_arr = np.array(y_pooled)

    # Pass 1: fit on all pooled data
    res1 = stats.linregress(x_arr, y_arr)
    y_pred1 = res1.intercept + res1.slope * x_arr
    residuals = y_arr - y_pred1
    raw_batch_rmse = float(np.sqrt(np.mean(residuals**2)))
    raw_batch_r2 = float(res1.rvalue**2) if res1.rvalue is not None else np.nan

    # Macro outlier detection: IQR on absolute residuals
    macro_outlier_mask = _detect_residual_outliers_iqr(residuals, whis=iqr_whis)
    n_macro_outliers = int(np.sum(macro_outlier_mask))

    # Pass 2: refit on inliers (need at least 2)
    inlier_mask = ~macro_outlier_mask
    n_inliers = int(np.sum(inlier_mask))
    if n_inliers < 2:
        return MacroRegressionResult(
            slope=res1.slope,
            intercept=res1.intercept,
            raw_slope=res1.slope,
            raw_intercept=res1.intercept,
            raw_batch_rmse=raw_batch_rmse,
            raw_batch_r2=raw_batch_r2,
            clean_batch_rmse=raw_batch_rmse,
            clean_batch_r2=raw_batch_r2,
            n_points=len(x_pooled),
            n_sensors=sensors_contributing,
            n_macro_outliers=n_macro_outliers,
            macro_outlier_mask=macro_outlier_mask,
            x_pooled=x_arr,
            y_pooled=y_arr,
        )

    x_clean = x_arr[inlier_mask]
    y_clean = y_arr[inlier_mask]
    res2 = stats.linregress(x_clean, y_clean)
    y_pred2 = res2.intercept + res2.slope * x_clean
    clean_batch_rmse = float(np.sqrt(np.mean((y_clean - y_pred2) ** 2)))
    clean_batch_r2 = float(res2.rvalue**2) if res2.rvalue is not None else np.nan

    return MacroRegressionResult(
        slope=res2.slope,
        intercept=res2.intercept,
        raw_slope=res1.slope,
        raw_intercept=res1.intercept,
        raw_batch_rmse=raw_batch_rmse,
        raw_batch_r2=raw_batch_r2,
        clean_batch_rmse=clean_batch_rmse,
        clean_batch_r2=clean_batch_r2,
        n_points=len(x_pooled),
        n_sensors=sensors_contributing,
        n_macro_outliers=n_macro_outliers,
        macro_outlier_mask=macro_outlier_mask,
        x_pooled=x_arr,
        y_pooled=y_arr,
    )
