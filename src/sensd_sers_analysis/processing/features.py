"""
Basic scalar feature extraction from SERS wide-format DataFrames.

Provides robust, macro-level features (max, mean, integral) without
peak detection. Designed for noisy spectra where peak-based features
are not yet reliable.
"""

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid as scipy_trapezoid

from sensd_sers_analysis.data.io import get_raman_shift, get_signals_matrix


def extract_basic_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Extract robust scalar features from a wide-format SERS DataFrame.

    For each sample (row), computes three macro-level features across
    all Raman shift intensity columns (rs_*):

    - max_intensity: Maximum intensity value in the spectrum.
    - mean_intensity: Mean intensity value (average across wavenumbers).
    - integral_area: Area under the curve via trapezoidal integration,
      using the actual Raman shift values from column names as x-axis.

    Args:
        df_wide: Wide DataFrame from load_sers_data; rows = samples,
            columns = metadata (sensor_id, serotype, etc.) + rs_* intensity
            columns (e.g., rs_400.00, rs_401.00, ...).

    Returns:
        DataFrame containing all original metadata columns plus the three
        feature columns (max_intensity, mean_intensity, integral_area).
        Raman shift (rs_*) columns are excluded to keep the output lightweight.

    Raises:
        ValueError: If df_wide has no Raman intensity columns.

    Example:
        >>> from sensd_sers_analysis.data.io import load_sers_data
        >>> from sensd_sers_analysis.processing.features import extract_basic_features
        >>> df = load_sers_data("example_data/")
        >>> df_feat = extract_basic_features(df)
        >>> df_feat[["serotype", "max_intensity", "integral_area"]].head()
    """
    if df_wide.empty:
        return df_wide.copy()

    signals = get_signals_matrix(df_wide)
    raman_shift = get_raman_shift(df_wide)

    # Use nan-aware aggregations for robustness when signals have scattered NaN
    # (e.g. from concatenated DataFrames with different Raman shift grids)
    max_intensity = np.nanmax(signals, axis=1)
    mean_intensity = np.nanmean(signals, axis=1)

    # Area under curve: trapezoidal integration, with fallback to sum for robustness
    if len(raman_shift) >= 2:
        integral_area = scipy_trapezoid(signals, x=raman_shift, axis=1)
        if np.any(np.isnan(integral_area)):
            integral_area = np.nansum(signals, axis=1).astype(float)
    else:
        integral_area = np.nansum(signals, axis=1).astype(float)

    metadata_cols = [
        c for c in df_wide.columns if not (isinstance(c, str) and c.startswith("rs_"))
    ]
    out = df_wide[metadata_cols].copy()
    out["max_intensity"] = max_intensity.astype(float)
    out["mean_intensity"] = mean_intensity.astype(float)
    out["integral_area"] = integral_area.astype(float)

    return out
