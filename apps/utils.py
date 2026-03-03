"""
App-level utilities for SERS Data Explorer.
"""

import pandas as pd

from sensd_sers_analysis.processing import (
    BASIC_FEATURE_COLUMNS,
    get_peak_height_columns,
    order_features_by_preference,
)


def get_available_feature_columns(
    df: pd.DataFrame,
    peak_infos_by_serotype: dict,
) -> list[str]:
    """
    Return ordered feature columns available in df (basic + dynamic peaks).

    Args:
        df: DataFrame with feature columns.
        peak_infos_by_serotype: Dict of serotype -> list of PeakWindowInfo.

    Returns:
        List of column names in preferred display order.
    """
    basic = [c for c in BASIC_FEATURE_COLUMNS if c in df.columns]
    peak_infos = next(iter(peak_infos_by_serotype.values()), [])
    peak_cols = [c for c in get_peak_height_columns(peak_infos) if c in df.columns]
    return order_features_by_preference(basic + peak_cols)
