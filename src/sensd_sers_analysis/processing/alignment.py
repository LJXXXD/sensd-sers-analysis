"""
Raman shift alignment and spectral window trimming.

Provides trim_raman_shift to enforce a uniform spectral window before
feature extraction and plotting, avoiding spurious peaks in non-overlapping
regions across sensors with differing wavelength ranges.
"""

from typing import Optional

import pandas as pd

from sensd_sers_analysis.data import RS_COL_PREFIX


def trim_raman_shift(
    wide_df: pd.DataFrame,
    min_shift: Optional[float] = None,
    max_shift: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter Raman shift columns to a specified spectral window.

    Drops rs_* intensity columns whose wavenumber falls outside
    [min_shift, max_shift]. All metadata columns are preserved. If both
    bounds are None, the DataFrame is returned unchanged.

    Args:
        wide_df: Wide-format DataFrame with metadata + rs_* intensity columns.
        min_shift: Lower bound (cm⁻¹) for Raman shift; columns with value
            below this are dropped. None = no lower bound.
        max_shift: Upper bound (cm⁻¹) for Raman shift; columns with value
            above this are dropped. None = no upper bound.

    Returns:
        DataFrame with same structure; only rs_* columns within the bounds
        are retained. Metadata columns unchanged.

    Example:
        >>> wide = load_sers_data("example_data/")
        >>> trimmed = trim_raman_shift(wide, min_shift=400, max_shift=1800)
        >>> assert (get_raman_shift(trimmed) >= 400).all()
        >>> assert (get_raman_shift(trimmed) <= 1800).all()
    """
    if wide_df.empty:
        return wide_df.copy()

    if min_shift is None and max_shift is None:
        return wide_df.copy()

    rs_cols = [
        c for c in wide_df.columns if isinstance(c, str) and c.startswith(RS_COL_PREFIX)
    ]
    if not rs_cols:
        return wide_df.copy()

    keep_cols: list[str] = []
    for col in rs_cols:
        try:
            val = float(col[len(RS_COL_PREFIX) :])
        except (ValueError, TypeError):
            keep_cols.append(col)
            continue
        if min_shift is not None and val < min_shift:
            continue
        if max_shift is not None and val > max_shift:
            continue
        keep_cols.append(col)

    metadata_cols = [c for c in wide_df.columns if c not in rs_cols]
    out_cols = metadata_cols + keep_cols
    return wide_df[[c for c in out_cols if c in wide_df.columns]].copy()
