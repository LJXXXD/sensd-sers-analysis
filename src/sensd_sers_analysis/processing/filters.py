"""
Filter discovery and application for SERS metadata.

Provides utilities to discover filterable metadata columns from a DataFrame
and apply include/exclude filters. Supports dynamic metadata (e.g. temperature,
notes) by inferring filter columns from the data rather than hardcoding.
"""

import pandas as pd

from sensd_sers_analysis.utils.natural_sort import order_concentration_labels
from sensd_sers_analysis.data import RS_COL_PREFIX

# Preferred display order for plot hue/style columns (subset of filter order).
DEFAULT_PLOT_HUE_ORDER = [
    "concentration_group",
    "serotype",
    "sensor_model",
    "date",
    "operator",
    "concentration",
    "log_concentration",
    "sensor_id",
    "test_id",
    "connection_id",
    "filename",
]

# Preferred display order for known metadata. Extras are appended.
DEFAULT_FILTER_ORDER = [
    "serotype",
    "concentration_group",
    "date",
    "sensor_id",
    "test_id",
    "sensor_model",
    "operator",
    "filename",
    "connection_id",
]

# Columns we never use as filters (spectral data)
NON_FILTER_COLS = {"raman_shift", "intensity"}


def _filter_mask(
    df: pd.DataFrame,
    col: str,
    selected: list | None,
    exclude: bool,
) -> pd.Series:
    """Apply include or exclude logic for one filter dimension."""
    if not selected or col not in df.columns:
        return pd.Series(True, index=df.index)
    in_set = df[col].astype(str).isin(selected)
    return ~in_set if exclude else in_set


def get_filter_options(
    df: pd.DataFrame,
    columns: list[str],
    filter_state: dict[str, tuple[list | None, bool]],
) -> dict[str, list]:
    """
    Compute cascading filter options for given columns.

    For each column, options are computed from rows that pass all previous
    filters. concentration_group options use natural sort order (Unknown last).

    Args:
        df: DataFrame with metadata.
        columns: Ordered list of column names to compute options for.
        filter_state: Dict mapping col -> (selected_values, exclude_bool).

    Returns:
        Dict mapping col -> list of option values (strings).
    """
    mask = pd.Series(True, index=df.index)
    result: dict[str, list] = {}

    def _str_opts(col: str) -> list:
        if col not in df.columns:
            return []
        series = df.loc[mask, col].dropna().astype(str)
        raw_vals = [v for v in series.unique() if v != ""]
        if col == "concentration_group" and raw_vals:
            return order_concentration_labels(raw_vals)
        return sorted(raw_vals)

    for col in columns:
        result[col] = _str_opts(col)
        selected, exclude = filter_state.get(col, (None, False))
        mask = mask & _filter_mask(df, col, selected, exclude)

    return result


def filter_sers_data(
    df: pd.DataFrame,
    filter_state: dict[str, tuple[list | None, bool]],
) -> pd.DataFrame:
    """
    Apply filters via a compound boolean mask.

    Args:
        df: DataFrame to filter.
        filter_state: Dict mapping col -> (selected_values, exclude_bool).
                     Empty/None selected means no filter for that col.

    Returns:
        Filtered DataFrame (view, not copy).
    """
    mask = pd.Series(True, index=df.index)
    for col, (selected, exclude) in filter_state.items():
        mask = mask & _filter_mask(df, col, selected, exclude)
    return df.loc[mask]


def get_filterable_columns(df) -> list[str]:
    """
    Discover metadata columns suitable for filtering, in display order.

    Excludes spectral columns (raman_shift, intensity, rs_*). Returns known
    metadata first (in DEFAULT_FILTER_ORDER), then any extra columns present
    in the DataFrame (e.g. temperature, notes) for future extensibility.

    Args:
        df: DataFrame with metadata columns.

    Returns:
        List of column names in preferred display order.
    """
    all_cols = set(df.columns)
    # Exclude spectral / non-metadata
    exclude = NON_FILTER_COLS.copy()
    for c in all_cols:
        if isinstance(c, str) and c.startswith(RS_COL_PREFIX):
            exclude.add(c)
    filterable = [c for c in all_cols if c not in exclude]

    ordered = [c for c in DEFAULT_FILTER_ORDER if c in filterable]
    extras = sorted(c for c in filterable if c not in DEFAULT_FILTER_ORDER)
    return ordered + extras


def get_plot_hue_columns(df: pd.DataFrame) -> list[str]:
    """
    Get metadata columns suitable for plot hue/style, in preferred order.

    Excludes spectral columns. Returns known columns first (DEFAULT_PLOT_HUE_ORDER),
    then extras present in the DataFrame.

    Args:
        df: DataFrame with metadata (tidy or wide).

    Returns:
        List of column names for hue/style dropdowns.
    """
    all_cols = set(df.columns)
    exclude = NON_FILTER_COLS.copy()
    for c in all_cols:
        if isinstance(c, str) and c.startswith(RS_COL_PREFIX):
            exclude.add(c)
    plotable = [c for c in all_cols if c not in exclude]

    ordered = [c for c in DEFAULT_PLOT_HUE_ORDER if c in plotable]
    extras = sorted(c for c in plotable if c not in DEFAULT_PLOT_HUE_ORDER)
    return ordered + extras


def get_feature_metadata_columns(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> list[str]:
    """
    Get non-feature columns from a feature DataFrame (for x-axis, hue in stats plots).

    Args:
        df: DataFrame from extract_basic_features (metadata + feature columns).
        feature_cols: Feature column names to exclude. If None, uses BASIC_FEATURE_COLUMNS.

    Returns:
        List of metadata column names present in df.
    """
    from sensd_sers_analysis.processing.features import BASIC_FEATURE_COLUMNS

    exclude = set(feature_cols or BASIC_FEATURE_COLUMNS)
    return [c for c in df.columns if c not in exclude]


def pick_preferred_column(
    available: list[str],
    preferred: tuple[str, ...] = ("concentration_group", "serotype"),
) -> str | None:
    """
    Pick the first preferred column that exists in available, or None.

    Args:
        available: List of available column names.
        preferred: Tuple of column names in order of preference.

    Returns:
        First preferred column in available, or None if none match.
    """
    for col in preferred:
        if col in available:
            return col
    return None
