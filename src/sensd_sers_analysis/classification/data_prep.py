"""
Identity eligibility and separate retrospective response screening.

Filters to Pass sensors only, drops outlier-flagged points from intra-sensor
regression, and assigns ``(N + 1)``-class targets: ``N`` serotypes observed on
bacterial samples plus the explicitly identified **Rinsate** controls.
"""

from typing import Optional

import pandas as pd

from sensd_sers_analysis.assessment import (
    fit_concentration_regression_cleaned,
    get_global_model_consistency_qa,
)
from sensd_sers_analysis.processing.metadata import sample_type_masks


def label_classification_dataset(
    df: pd.DataFrame, *, serotype_col: str = "serotype"
) -> pd.DataFrame:
    """Copy identity-eligible rows without response-based screening.

    Explicit sample type supplies control identity; bacterial rows need a usable
    serotype. Original source indices and order are retained. Cohort QA and
    outlier diagnostics do not determine eligibility for held-out evaluation.
    """
    rinsate, bacteria = sample_type_masks(df)
    labels = pd.Series("Unknown", index=df.index, dtype=object)
    labels.loc[rinsate] = "Rinsate"
    if serotype_col in df:
        serotypes = df[serotype_col].astype("string").str.strip()
        valid = (
            serotypes.notna()
            & serotypes.ne("")
            & ~serotypes.str.lower().isin(["nan", "none", "unknown"])
        )
        labels.loc[bacteria & valid] = serotypes.loc[bacteria & valid]
    out = df.loc[labels.ne("Unknown")].copy()
    out["target"] = labels.loc[out.index]
    return out


def prepare_classification_dataset(
    df: pd.DataFrame,
    *,
    excluded_map: Optional[dict[tuple[str, str], set[str]]] = None,
    feature_cols: Optional[list[str]] = None,
    inlier_feature: str = "integral_area",
    sensor_col: str = "sensor_id",
    serotype_col: str = "serotype",
    log_conc_col: str = "log_concentration",
    concentration_group_col: str = "concentration_group",
    concentration_col: str = "concentration",
) -> pd.DataFrame:
    """
    Produce strictly clean data for serotype classification.

    - Only rows from sensors marked Pass in Global Assessment (integral_area).
    - Re-runs intra-sensor outlier detection on integral_area; keeps inliers only.
    - **Rinsate** only for explicit ``sample_type = Rinsate control``.
    - Bacteria sample rows: ``target`` is the row's ``serotype`` string; rows with
      missing or unusable serotype labels are dropped.

    Args:
        df: Filtered feature DataFrame after preprocessing and QA policy inputs.
        excluded_map: {(serotype, feature): {sensor_id, ...}}. If None, runs QA.
        feature_cols: Features for QA when excluded_map is None.
        inlier_feature: Feature for outlier removal (integral_area).
        sensor_col: Sensor identifier column.
        serotype_col: Serotype column.
        log_conc_col: Log concentration column.
        concentration_group_col: Column for 0 CFU detection.
        concentration_col: Raw concentration column (used when available).

    Returns:
        DataFrame with ``target`` column (serotype names on positive CFU,
        ``"Rinsate"`` on explicit controls). No merges; no dropna on feature columns.
    """
    required = [sensor_col, serotype_col, concentration_group_col]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    if inlier_feature not in df.columns:
        return pd.DataFrame()

    if excluded_map is None:
        feat_cols = feature_cols or [inlier_feature]
        _, excluded_map = get_global_model_consistency_qa(
            df, feature_cols=[c for c in feat_cols if c in df.columns]
        )

    # 1. Identify Pass sensors for integral_area (per serotype)
    all_sensors = set(df[sensor_col].dropna().astype(str).unique())
    keep_indices: set[int] = set()

    for sero in df[serotype_col].dropna().astype(str).unique():
        excluded = excluded_map.get((sero, inlier_feature), set())
        pass_sensors = all_sensors - excluded
        if not pass_sensors:
            continue

        subset = df[
            (df[serotype_col].astype(str) == str(sero))
            & (df[sensor_col].astype(str).isin(pass_sensors))
        ].copy()
        if subset.empty:
            continue

        rinsate_mask, bacteria_mask = sample_type_masks(subset)
        for idx in subset.index[rinsate_mask]:
            keep_indices.add(idx)

        # Fit only loggable bacterial readings; retain non-loggable sample identities.
        pos_mask = bacteria_mask
        subset_pos = subset.loc[pos_mask]
        if subset_pos.empty:
            continue

        cres = fit_concentration_regression_cleaned(
            subset_pos, inlier_feature, log_conc_col=log_conc_col
        )
        if cres is None:
            for idx in subset_pos.index:
                keep_indices.add(idx)
            continue

        valid = subset_pos[[log_conc_col, inlier_feature]].notna().all(axis=1)
        # Non-loggable bacterial samples remain bacteria, including zero counts.
        keep_indices.update(subset_pos.index[~valid])
        sub_fit = subset_pos.loc[valid]
        inlier_mask = ~cres.outlier_mask
        for idx in sub_fit.index[inlier_mask]:
            keep_indices.add(idx)

    if not keep_indices:
        return pd.DataFrame()

    out = df.loc[df.index.isin(keep_indices)].copy()

    # Sample identity defines the class independently of measured concentration.
    out["target"] = "Unknown"
    rinsate_mask, bacteria_mask = sample_type_masks(out)
    out.loc[rinsate_mask, "target"] = "Rinsate"

    pos_unknown = (out["target"] == "Unknown") & bacteria_mask
    if serotype_col in out.columns:
        sero_str = out[serotype_col].astype(str).str.strip()
        valid_sero = sero_str.notna() & (sero_str != "") & ~sero_str.str.lower().eq("nan")
        use = pos_unknown & valid_sero
        out.loc[use, "target"] = sero_str.loc[use].values
    out = out[out["target"] != "Unknown"].copy()
    return out
