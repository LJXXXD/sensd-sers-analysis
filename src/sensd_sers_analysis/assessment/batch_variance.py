"""
Batch variance analysis for multi-sensor system stability.

Evaluates variance across sensors of the same design (grouped by sensor_id)
to identify sensors that deviate significantly from the population.
"""

from typing import Optional

import numpy as np
import pandas as pd


def compute_batch_variance(
    df: pd.DataFrame,
    feature_col: str,
    *,
    sensor_col: str = "sensor_id",
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute per-sensor summary and deviation from batch mean.

    Args:
        df: Feature DataFrame.
        feature_col: Feature to analyze.
        sensor_col: Column identifying individual sensors.
        group_cols: Additional grouping (e.g., concentration_group) to
            compute variance within each group.

    Returns:
        DataFrame with columns: [sensor_col, group_cols (if any), feature,
        n_samples, mean, std, cv, batch_mean, batch_std, z_from_batch,
        deviation_pct]. z_from_batch is (mean - batch_mean) / batch_std;
        deviation_pct is (mean - batch_mean) / batch_mean * 100.
    """
    if feature_col not in df.columns:
        raise ValueError(
            f"feature_col '{feature_col}' not in DataFrame. "
            f"Available: {list(df.columns)}"
        )
    if sensor_col not in df.columns:
        raise ValueError(
            f"sensor_col '{sensor_col}' not in DataFrame. Available: {list(df.columns)}"
        )

    id_cols = [sensor_col]
    if group_cols:
        id_cols = id_cols + list(group_cols)
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"group_cols {missing} not in DataFrame. Available: {list(df.columns)}"
            )

    agg = (
        df.groupby(id_cols, dropna=False)
        .agg(
            n_samples=(feature_col, "count"),
            mean=(feature_col, "mean"),
            std=(feature_col, "std"),
        )
        .reset_index()
    )

    agg["cv"] = np.where(
        agg["mean"] != 0,
        agg["std"] / np.abs(agg["mean"]),
        np.nan,
    )

    # Batch-level stats (over all sensors in same group)
    if group_cols:
        batch = (
            agg.groupby(group_cols, dropna=False)
            .agg(
                batch_mean=("mean", "mean"),
                batch_std=("mean", "std"),
            )
            .reset_index()
        )
        agg = agg.merge(batch, on=group_cols, how="left")
    else:
        agg["batch_mean"] = agg["mean"].mean()
        agg["batch_std"] = agg["mean"].std()

    batch_std_safe = np.where(
        np.isfinite(agg["batch_std"]) & (agg["batch_std"] > 0),
        agg["batch_std"],
        np.nan,
    )
    agg["z_from_batch"] = (agg["mean"] - agg["batch_mean"]) / batch_std_safe

    batch_mean_safe = np.where(
        np.isfinite(agg["batch_mean"]) & (agg["batch_mean"] != 0),
        agg["batch_mean"],
        np.nan,
    )
    agg["deviation_pct"] = (agg["mean"] - agg["batch_mean"]) / batch_mean_safe * 100

    agg["feature"] = feature_col
    return agg


def identify_deviating_sensors(
    batch_df: pd.DataFrame,
    *,
    z_threshold: float = 2.0,
    sensor_col: str = "sensor_id",
) -> pd.DataFrame:
    """
    Flag sensors that deviate significantly from the batch.

    Args:
        batch_df: Output of compute_batch_variance.
        z_threshold: |z_from_batch| > this indicates deviation.
        sensor_col: Sensor identifier column.

    Returns:
        Subset of batch_df where |z_from_batch| > z_threshold, sorted by
        |z_from_batch| descending.
    """
    if "z_from_batch" not in batch_df.columns:
        return pd.DataFrame()

    mask = np.isfinite(batch_df["z_from_batch"]) & (
        np.abs(batch_df["z_from_batch"]) > z_threshold
    )
    out = batch_df.loc[mask].copy()
    out["abs_z"] = np.abs(out["z_from_batch"])
    return out.sort_values("abs_z", ascending=False).drop(columns=["abs_z"])
