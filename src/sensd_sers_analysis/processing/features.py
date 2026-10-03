"""
Basic scalar feature extraction from SERS wide-format DataFrames.

Provides robust, macro-level features (max, mean, integral) and PCA
components (PC1, PC2) for assessment. Designed for noisy spectra where
peak-based features are not yet reliable.
"""

import logging

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid as scipy_trapezoid

from sensd_sers_analysis.data import RS_COL_PREFIX, get_raman_shift, get_signals_matrix
from sensd_sers_analysis.processing.pca_features import add_pca_features
from sensd_sers_analysis.processing.targeted_peak_features import (
    list_targeted_peak_feature_columns,
)

logger = logging.getLogger(__name__)

# Column names produced by extract_basic_features; use for stats plots and validation.
BASIC_FEATURE_COLUMNS = [
    "max_intensity",
    "mean_intensity",
    "integral_area",
    "PC1",
    "PC2",
]

# Preferred display and default order for feature dropdowns and tables.
PREFERRED_FEATURE_ORDER = [
    "integral_area",
    "mean_intensity",
    "max_intensity",
    "peak_near_501_8",
    "peak_near_613_7",
    "peak_near_809_7",
    "peak_near_1066_5",
    "peak_near_1196_8",
    "PC1",
    "PC2",
]

# Default features for Global QA Table multiselect.
DEFAULT_GLOBAL_QA_FEATURES = [
    "integral_area",
    "peak_near_501_8",
    "peak_near_809_7",
    "peak_near_1066_5",
]

# Base feature columns for serotype classification (ML input).
CLASSIFICATION_FEATURE_BASE = [
    "integral_area",
    "max_intensity",
    "mean_intensity",
]


def get_available_feature_columns(
    df: pd.DataFrame,
    peak_infos_by_serotype: dict | None = None,
) -> list[str]:
    """
    Return ordered feature columns available in ``df`` (basic + targeted peaks).

    Parameters
    ----------
    df:
        DataFrame with feature columns (including ``peak_near_*`` when present).
    peak_infos_by_serotype:
        Unused; retained for backward-compatible call sites.

    Returns
    -------
    list[str]
        Column names in preferred display order.
    """
    del peak_infos_by_serotype
    basic = [c for c in BASIC_FEATURE_COLUMNS if c in df.columns]
    targeted = list_targeted_peak_feature_columns(df.columns)
    targeted = [c for c in targeted if c in df.columns]
    return order_features_by_preference(basic + targeted)


def order_features_by_preference(
    features: list[str],
    *,
    preference: list[str] | None = None,
) -> list[str]:
    """Order feature list by PREFERRED_FEATURE_ORDER; extras at end."""
    pref = preference or PREFERRED_FEATURE_ORDER
    seen = set(pref)
    ordered = [f for f in pref if f in features]
    extras = [f for f in features if f not in seen]
    return ordered + sorted(extras)


def extract_basic_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Extract scalar features and exploratory PCA scores from wide SERS data.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Samples as rows, with metadata and ``rs_*`` intensity columns. Raman
        coordinates are read from column names and sorted numerically.

    Returns
    -------
    pd.DataFrame
        Metadata, maximum and mean intensity, trapezoidal integral area, and
        available PCA columns, retaining the input index. Intensity columns
        are excluded. Empty input is returned as an unchanged copy.

    Raises
    ------
    ValueError
        If nonempty input has no Raman intensity columns.

    Notes
    -----
    Maximum and mean ignore missing intensities. Integral area uses the full
    coordinate grid and has units of intensity times Raman-shift units. An
    integral with nonfinite values or fewer than two coordinates is unavailable
    (NaN), with a logged warning; it is not imputed or replaced by a sum.
    Unavailable integrals do not affect other rows. PCA is fitted to the supplied
    dataset for exploration; these scores are not training-only transforms.
    """
    if df_wide.empty:
        return df_wide.copy()

    signals = get_signals_matrix(df_wide)
    raman_shift = get_raman_shift(df_wide)

    # Use nan-aware aggregations for robustness when signals have scattered NaN
    # (e.g. from concatenated DataFrames with different Raman shift grids)
    max_intensity = np.nanmax(signals, axis=1)
    mean_intensity = np.nanmean(signals, axis=1)

    if len(raman_shift) >= 2:
        integral_area = scipy_trapezoid(signals, x=raman_shift, axis=1)
    else:
        integral_area = np.full(len(df_wide), np.nan)

    unavailable = ~np.isfinite(integral_area)
    if np.any(unavailable):
        integral_area[unavailable] = np.nan
        logger.warning(
            "Integral area unavailable for %d spectra; finite values and at least "
            "two Raman coordinates are required.",
            np.count_nonzero(unavailable),
        )

    metadata_cols = [
        c for c in df_wide.columns if not (isinstance(c, str) and c.startswith(RS_COL_PREFIX))
    ]
    out = df_wide[metadata_cols].copy()
    out["max_intensity"] = max_intensity.astype(float)
    out["mean_intensity"] = mean_intensity.astype(float)
    out["integral_area"] = integral_area.astype(float)

    # Add PCA features (PC1, PC2, variance ratios)
    pca_df = add_pca_features(df_wide)
    if not pca_df.empty:
        out = out.join(pca_df[["PC1", "PC2", "PC1_var_ratio", "PC2_var_ratio"]], how="left")

    return out
