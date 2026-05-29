"""
Clean tabular data for concentration regression (positive CFU, dynamic serotypes).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from sensd_sers_analysis.classification.data_prep import prepare_classification_dataset
from sensd_sers_analysis.processing import extract_scalar_concentration


def prepare_concentration_regression_data(
    df: pd.DataFrame,
    *,
    excluded_map: Optional[dict[tuple[str, str], set[str]]] = None,
    feature_cols: Optional[list[str]] = None,
    inlier_feature: str = "integral_area",
    sensor_col: str = "sensor_id",
    serotype_col: str = "serotype",
    log_conc_col: str = "log_concentration",
    concentration_col: str = "concentration",
    concentration_group_col: str = "concentration_group",
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Build strictly clean rows for log10 concentration regression.

    Reuses the same Pass-sensor and inlier cleaning as serotype classification
    (including serotype / Rinsate labeling). Retains only **non-Rinsate** rows
    with **positive** concentration and finite ``log_concentration`` (one
    row-level serotype label per sample).

    Parameters
    ----------
    df:
        Filtered feature dataframe after preprocessing and QA inputs.
    excluded_map, feature_cols, inlier_feature:
        Passed through to :func:`~sensd_sers_analysis.classification.prepare_classification_dataset`.
    sensor_col, serotype_col, log_conc_col, concentration_col,
    concentration_group_col, target_col:
        Column names.

    Returns
    -------
    pd.DataFrame
        Copy with ``target`` equal to the serotype string on each positive-CFU row
        and valid regression target. Empty if prerequisites are missing.
    """
    classification_clean = prepare_classification_dataset(
        df,
        excluded_map=excluded_map,
        feature_cols=feature_cols,
        inlier_feature=inlier_feature,
        sensor_col=sensor_col,
        serotype_col=serotype_col,
        log_conc_col=log_conc_col,
        concentration_group_col=concentration_group_col,
        concentration_col=concentration_col,
    )
    if classification_clean.empty:
        return pd.DataFrame()

    if log_conc_col not in classification_clean.columns:
        return pd.DataFrame()

    out = classification_clean[classification_clean[target_col].astype(str) != "Rinsate"].copy()
    out = out[out[log_conc_col].notna()].copy()

    if concentration_col in out.columns:
        conc = extract_scalar_concentration(out[concentration_col], out)
        out = out.loc[conc.notna() & (conc > 0)].copy()
    else:
        out = out[out[concentration_group_col].astype(str) != "0 CFU"].copy()

    if out.empty:
        return pd.DataFrame()

    out = out.reset_index(drop=True)
    return out
