"""
Clean tabular data for concentration regression (positive CFU, dynamic serotypes).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from sensd_sers_analysis.classification.data_prep import (
    label_classification_dataset,
    prepare_classification_dataset,
)
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
    apply_response_screening: bool = True,
) -> pd.DataFrame:
    """
    Build finite positive actual-CFU rows for log10 concentration regression.

    With ``apply_response_screening=True``, reuses retrospective Pass-sensor and inlier cleaning
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
    apply_response_screening:
        Retrospective diagnostic cleaning. Model services pass False so held-out
        responses never determine the evaluated cohort.

    Returns
    -------
    pd.DataFrame
        Copy with ``target_col`` equal to the serotype string on each finite,
        positive-CFU row and finite regression target. Source indices and row
        order are retained for prediction alignment. Empty if prerequisites are missing.

    Raises
    ------
    ValueError
        If a custom ``target_col`` would overwrite another input column.
    """
    classification_clean = (
        prepare_classification_dataset(
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
        if apply_response_screening
        else label_classification_dataset(df, serotype_col=serotype_col)
    )
    if classification_clean.empty:
        return pd.DataFrame()

    if any(c not in classification_clean.columns for c in (log_conc_col, concentration_col)):
        return pd.DataFrame()

    if target_col != "target":
        if target_col in classification_clean.columns:
            raise ValueError(f"Class target column {target_col!r} already exists.")
        classification_clean = classification_clean.rename(columns={"target": target_col})

    out = classification_clean[classification_clean[target_col].astype(str) != "Rinsate"].copy()
    conc = extract_scalar_concentration(out[concentration_col], out)
    valid = np.isfinite(out[log_conc_col]) & np.isfinite(conc) & (conc > 0)
    out = out.loc[valid].copy()

    if out.empty:
        return pd.DataFrame()

    return out
