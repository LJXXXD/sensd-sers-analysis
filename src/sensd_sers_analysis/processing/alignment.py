"""
Raman shift alignment and spectral window trimming.

Provides ``trim_raman_shift`` to enforce a uniform spectral window before
feature extraction and plotting, and ``snap_spectra_to_master_grid`` to align
heterogeneous sensor grids onto a session master axis via overlap-localized
linear interpolation (non-overlapping regions remain NaN).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from sensd_sers_analysis.config import SNAP_SHIFT_DEDUPE_ATOL, SNAP_SHIFT_DEDUPE_RTOL
from sensd_sers_analysis.data import RS_COL_PREFIX

logger = logging.getLogger(__name__)


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

    rs_cols = [c for c in wide_df.columns if isinstance(c, str) and c.startswith(RS_COL_PREFIX)]
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


def _parse_rs_shift(column: str) -> float | None:
    if not isinstance(column, str) or not column.startswith(RS_COL_PREFIX):
        return None
    try:
        return float(column[len(RS_COL_PREFIX) :])
    except (TypeError, ValueError):
        return None


def _merge_sorted_near_duplicate_shifts(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge consecutive sorted shifts that are numerically coincident.

    Representative shift is the mean of the group; intensity is ``nanmean``.
    """
    if x.size == 0:
        return x.astype(float), y.astype(float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out_x: list[float] = []
    out_y: list[float] = []
    i = 0
    n = x.size
    while i < n:
        j = i + 1
        group_idx = [i]
        while j < n and np.isclose(x[j], x[i], rtol=rtol, atol=atol):
            group_idx.append(j)
            j += 1
        out_x.append(float(np.mean(x[group_idx])))
        out_y.append(float(np.nanmean(y[group_idx])))
        i = j
    return np.asarray(out_x, dtype=float), np.asarray(out_y, dtype=float)


def _row_sorted_xy(
    row: pd.Series,
    parseable_rs_cols: list[str],
    *,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite (shift, intensity) pairs for one row, sorted by shift, merged."""
    xs: list[float] = []
    ys: list[float] = []
    for col in parseable_rs_cols:
        val = row[col]
        if pd.isna(val):
            continue
        parsed = _parse_rs_shift(col)
        if parsed is None:
            continue
        xs.append(float(parsed))
        ys.append(float(val))
    if not xs:
        return np.array([]), np.array([])
    order = np.argsort(np.asarray(xs), kind="mergesort")
    x_arr = np.asarray(xs, dtype=float)[order]
    y_arr = np.asarray(ys, dtype=float)[order]
    return _merge_sorted_near_duplicate_shifts(x_arr, y_arr, rtol=rtol, atol=atol)


def _row_comprehensiveness_score(x: np.ndarray) -> tuple[float, float, int]:
    """
    Score one spectrum's native grid for master selection.

    Lexicographic order (all maximize): spectral span, point density, count.
    """
    n = int(x.size)
    if n < 2:
        return (0.0, 0.0, n)
    span = float(x[-1] - x[0])
    if span <= 0.0:
        return (0.0, 0.0, n)
    density = float(n) / span
    return (span, density, n)


def snap_spectra_to_master_grid(
    wide_df: pd.DataFrame,
    *,
    dedupe_rtol: float = SNAP_SHIFT_DEDUPE_RTOL,
    dedupe_atol: float = SNAP_SHIFT_DEDUPE_ATOL,
) -> pd.DataFrame:
    """
    Align all spectra onto a single Raman-shift axis using a session master grid.

    The master grid is taken from the spectrum (row) with the largest
    spectral span; ties break on point density (points per cm⁻¹), then point
    count. Each row is linearly interpolated onto master shifts that lie
    strictly within that row's native ``[min_shift, max_shift]`` support;
    master coordinates outside that interval remain NaN (no extrapolation).

    Non-numeric ``rs_*`` column names are passed through unchanged after the
    snapped spectral block.

    Parameters
    ----------
    wide_df:
        Wide-format DataFrame (e.g. ``pd.concat`` of per-file loads) with
        metadata and ``rs_*`` intensity columns.
    dedupe_rtol, dedupe_atol:
        Passed to :func:`numpy.isclose` when merging nearly duplicate shifts
        on a single row before interpolation.

    Returns
    -------
    pd.DataFrame
        Same row index and metadata columns; ``rs_*`` columns are the master
        grid only, dense for overlapping regions and NaN elsewhere.
    """
    if wide_df.empty:
        return wide_df.copy()

    parseable: list[str] = []
    unparseable: list[str] = []
    for col in wide_df.columns:
        if isinstance(col, str) and col.startswith(RS_COL_PREFIX):
            if _parse_rs_shift(col) is not None:
                parseable.append(col)
            else:
                unparseable.append(col)

    if not parseable:
        return wide_df.copy()

    meta_cols = [c for c in wide_df.columns if c not in set(parseable + unparseable)]
    n_rows = len(wide_df)

    best_idx = 0
    best_score = (-1.0, -1.0, -1)
    for i in range(n_rows):
        x_row, _ = _row_sorted_xy(wide_df.iloc[i], parseable, rtol=dedupe_rtol, atol=dedupe_atol)
        score = _row_comprehensiveness_score(x_row)
        if score > best_score:
            best_score = score
            best_idx = i

    x_master, _ = _row_sorted_xy(
        wide_df.iloc[best_idx], parseable, rtol=dedupe_rtol, atol=dedupe_atol
    )
    if x_master.size < 2:
        logger.warning(
            "Master grid snapping skipped: no row has at least two finite "
            "spectral points (best row index %s).",
            best_idx,
        )
        return wide_df.copy()

    # Fixed-width labels keep ``rs_*`` parseable for downstream tools and avoid
    # rare collisions from ``format_float_positional`` on near-tied shifts.
    master_cols = [f"{RS_COL_PREFIX}{float(v):.8f}" for v in x_master]

    spectral = np.full((n_rows, x_master.size), np.nan, dtype=float)

    for i in range(n_rows):
        x_native, y_native = _row_sorted_xy(
            wide_df.iloc[i], parseable, rtol=dedupe_rtol, atol=dedupe_atol
        )
        if x_native.size == 0:
            continue
        if x_native.size == 1:
            hits = np.isclose(x_master, x_native[0], rtol=dedupe_rtol, atol=dedupe_atol)
            spectral[i, hits] = y_native[0]
            continue
        lo, hi = float(x_native[0]), float(x_native[-1])
        overlap = (x_master >= lo) & (x_master <= hi)
        if not np.any(overlap):
            continue
        spectral[i, overlap] = np.interp(
            x_master[overlap], x_native, y_native, left=np.nan, right=np.nan
        )

    meta_df = wide_df[meta_cols]
    spec_df = pd.DataFrame(spectral, index=wide_df.index, columns=master_cols)
    parts: list[pd.DataFrame] = [meta_df, spec_df]
    if unparseable:
        parts.append(wide_df[unparseable])

    out = pd.concat(parts, axis=1)
    logger.info(
        "Master grid snapping: master_row_index=%s, n_master_points=%d, n_rows=%d",
        best_idx,
        x_master.size,
        n_rows,
    )
    return out.copy()
