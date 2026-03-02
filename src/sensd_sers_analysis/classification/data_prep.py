"""
Phase 2 data preparation: strictly clean data from Phase 1.

Filters to Pass sensors only, drops outlier-flagged points from intra-sensor
regression, and assigns 3-class target: ST, SE, Rinsate.
"""

from typing import Optional

import pandas as pd

from sensd_sers_analysis.assessment import (
    fit_concentration_regression_cleaned,
    get_global_model_consistency_qa,
)
from sensd_sers_analysis.processing.metadata import _extract_scalar_concentration


def prepare_phase2_data(
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
    Produce strictly clean data for Phase 2 classification.

    - Only rows from sensors marked Pass in Global Assessment (integral_area).
    - Re-runs intra-sensor outlier detection on integral_area; keeps inliers only.
    - Strict 3-class labeling: Rinsate (conc==0), ST, SE. Drops all others.

    Args:
        df: Filtered feature DataFrame from Phase 1.
        excluded_map: {(serotype, feature): {sensor_id, ...}}. If None, runs QA.
        feature_cols: Features for QA when excluded_map is None.
        inlier_feature: Feature for outlier removal (integral_area).
        sensor_col: Sensor identifier column.
        serotype_col: Serotype column.
        log_conc_col: Log concentration column.
        concentration_group_col: Column for 0 CFU detection.
        concentration_col: Raw concentration column (used when available).

    Returns:
        DataFrame with "target" column (ST, SE, Rinsate). No merges; no dropna.
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

    for (sero, feat), excluded in excluded_map.items():
        if feat != inlier_feature:
            continue
        pass_sensors = all_sensors - excluded
        if not pass_sensors:
            continue

        subset = df[
            (df[serotype_col].astype(str) == str(sero))
            & (df[sensor_col].astype(str).isin(pass_sensors))
        ].copy()
        if subset.empty:
            continue

        # 2. Rinsate: concentration == 0 or concentration_group == "0 CFU" (EXACT)
        if concentration_col in df.columns:
            conc = _extract_scalar_concentration(subset[concentration_col], subset)
            rinsate_mask = conc.notna() & (conc <= 0)
        else:
            rinsate_mask = subset[concentration_group_col].astype(str) == "0 CFU"
        for idx in subset.index[rinsate_mask]:
            keep_indices.add(idx)

        # 3. >0 CFU: run outlier detection, keep inliers only
        pos_mask = ~rinsate_mask
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
        sub_fit = subset_pos.loc[valid]
        inlier_mask = ~cres.outlier_mask
        for idx in sub_fit.index[inlier_mask]:
            keep_indices.add(idx)

    if not keep_indices:
        return pd.DataFrame()

    out = df.loc[list(keep_indices)].copy()

    # 4. Strict 3-class labeling: Rinsate (conc==0 or exact "0 CFU"), ST, SE
    out["target"] = "Unknown"
    if concentration_col in out.columns:
        conc = _extract_scalar_concentration(out[concentration_col], out)
        rinsate_mask = conc.notna() & (conc <= 0)
    else:
        rinsate_mask = out[concentration_group_col].astype(str) == "0 CFU"
    out.loc[rinsate_mask, "target"] = "Rinsate"
    st_mask = (out["target"] == "Unknown") & (out[serotype_col].astype(str) == "ST")
    se_mask = (out["target"] == "Unknown") & (out[serotype_col].astype(str) == "SE")
    out.loc[st_mask, "target"] = "ST"
    out.loc[se_mask, "target"] = "SE"
    out = out[out["target"] != "Unknown"].copy()
    return out
