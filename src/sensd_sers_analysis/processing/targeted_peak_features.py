"""
Fixed-anchor peak height features for SERS wide-format spectra.

For each target Raman shift, finds the maximum intensity within a symmetric
search window around the anchor (optional local baseline from window edges).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sensd_sers_analysis.config.targeted_peaks import TARGETED_PEAK_SEARCH_HALF_WIDTH_CM1
from sensd_sers_analysis.data import get_raman_shift, get_signals_matrix


def target_anchor_to_feature_name(anchor_cm1: float) -> str:
    """
    Build a stable column name from a target Raman shift (cm⁻¹).

    Uses a trimmed decimal string so that, for example, 501.8 becomes
    ``peak_near_501_8``.

    Parameters
    ----------
    anchor_cm1:
        Target Raman shift in cm⁻¹.

    Returns
    -------
    str
        Column name prefixed with ``peak_near_``.
    """

    text = f"{float(anchor_cm1):.6f}".rstrip("0").rstrip(".")
    return "peak_near_" + text.replace(".", "_").replace("-", "neg")


def parse_feature_name_to_anchor(column_name: str) -> float | None:
    """
    Recover the anchor wavenumber from a ``peak_near_*`` column name.

    Names are produced by :func:`target_anchor_to_feature_name` (for example
    ``peak_near_501_8`` encodes 501.8 cm⁻¹).

    Parameters
    ----------
    column_name:
        Dataframe column such as ``peak_near_501_8``.

    Returns
    -------
    float or None
        Parsed anchor, or None if the name does not match the convention.
    """

    prefix = "peak_near_"
    if not column_name.startswith(prefix):
        return None
    body = column_name[len(prefix) :].replace("neg", "-")
    if not body:
        return None
    if "_" not in body:
        try:
            return float(body)
        except ValueError:
            return None
    left, right = body.split("_", 1)
    try:
        if "_" in right:
            return float(f"{left}.{right.replace('_', '')}")
        return float(f"{left}.{right}")
    except ValueError:
        return None


def list_targeted_peak_feature_columns(columns: Sequence[str]) -> list[str]:
    """
    Return ``peak_near_*`` column names ordered by parsed anchor wavenumber.

    Parameters
    ----------
    columns:
        Iterable of dataframe column names.

    Returns
    -------
    list[str]
        Targeted peak feature columns sorted by inferred anchor.
    """

    scored: list[tuple[float, str]] = []
    for name in columns:
        if not isinstance(name, str) or not name.startswith("peak_near_"):
            continue
        anchor = parse_feature_name_to_anchor(name)
        if anchor is not None:
            scored.append((anchor, name))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [name for _, name in scored]


def extract_targeted_peak_height_features(
    df_wide: pd.DataFrame,
    anchor_cm1: Sequence[float],
    *,
    half_width_cm1: float = TARGETED_PEAK_SEARCH_HALF_WIDTH_CM1,
) -> pd.DataFrame:
    """
    Extract baseline-adjusted peak heights near fixed Raman-shift anchors.

    For each row (spectrum) and each anchor, the intensity maximum is taken
    on grid points with Raman shift in ``[anchor − half_width, anchor +
    half_width]``. A local baseline is estimated from the mean of the first
    and last 10% of samples inside that window (same edge policy as dynamic
    peak extraction when windows are wide enough).

    Parameters
    ----------
    df_wide:
        Wide-format dataframe with ``rs_*`` intensity columns.
    anchor_cm1:
        Target Raman shifts (cm⁻¹), typically user-defined anchors.
    half_width_cm1:
        Half-width of the search interval around each anchor.

    Returns
    -------
    pd.DataFrame
        One column per anchor (``peak_near_*``) aligned to ``df_wide.index``.
        Empty dataframe if ``df_wide`` is empty or there are no anchors.
    """

    if df_wide.empty or not anchor_cm1:
        return pd.DataFrame(index=df_wide.index)

    signals = get_signals_matrix(df_wide)
    raman_shift = np.asarray(get_raman_shift(df_wide), dtype=float)
    if raman_shift.size < 2 or signals.size == 0:
        return pd.DataFrame(index=df_wide.index)

    x = raman_shift
    n_samples = signals.shape[0]
    out: dict[str, np.ndarray] = {}

    for anchor in anchor_cm1:
        col = target_anchor_to_feature_name(float(anchor))
        mask = (x >= float(anchor) - half_width_cm1) & (x <= float(anchor) + half_width_cm1)
        if not np.any(mask):
            out[col] = np.full(n_samples, np.nan, dtype=float)
            continue

        window_y = signals[:, mask]
        n_edge = max(1, int(window_y.shape[1] * 0.1))
        left = window_y[:, :n_edge]
        right = window_y[:, -n_edge:]
        baseline = (np.nanmean(left, axis=1) + np.nanmean(right, axis=1)) / 2.0
        peak_height = np.nanmax(window_y, axis=1) - baseline
        out[col] = np.asarray(peak_height, dtype=float)

    return pd.DataFrame(out, index=df_wide.index)


def detect_targeted_peaks_on_spectrum_row(
    y_row: np.ndarray,
    raman_x: np.ndarray,
    anchor_cm1: Sequence[float],
    *,
    half_width_cm1: float = TARGETED_PEAK_SEARCH_HALF_WIDTH_CM1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find local maxima near anchors for a single spectrum row.

    Parameters
    ----------
    y_row:
        Intensity values aligned with ``raman_x``.
    raman_x:
        Raman-shift grid (cm⁻¹).
    anchor_cm1:
        Target anchors (cm⁻¹).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(detected_shifts, raw_heights, baseline_subtracted_heights)``, each
        of shape ``(len(anchor_cm1),)`` with NaNs when the window is empty.
    """

    y = np.asarray(y_row, dtype=float)
    x = np.asarray(raman_x, dtype=float)
    n = len(anchor_cm1)
    shifts = np.full(n, np.nan, dtype=float)
    raw_h = np.full(n, np.nan, dtype=float)
    adj_h = np.full(n, np.nan, dtype=float)
    if x.size == 0 or y.size != x.size:
        return shifts, raw_h, adj_h

    for i, anchor in enumerate(anchor_cm1):
        mask = (x >= float(anchor) - half_width_cm1) & (x <= float(anchor) + half_width_cm1)
        if not np.any(mask):
            continue
        wx = x[mask]
        wy = y[mask]
        if not np.any(np.isfinite(wy)):
            continue
        n_edge = max(1, int(len(wy) * 0.1))
        left = wy[:n_edge]
        right = wy[-n_edge:]
        baseline = (np.nanmean(left) + np.nanmean(right)) / 2.0
        j = int(np.nanargmax(wy))
        shifts[i] = float(wx[j])
        raw_h[i] = float(wy[j])
        adj_h[i] = float(wy[j]) - float(baseline)
    return shifts, raw_h, adj_h


def compute_targeted_peak_positions_on_mean(
    mean_spectrum: np.ndarray,
    raman_x: np.ndarray,
    anchor_cm1: Sequence[float],
    *,
    half_width_cm1: float = TARGETED_PEAK_SEARCH_HALF_WIDTH_CM1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Locate peak Raman shifts on a mean spectrum for each anchor window.

    Parameters
    ----------
    mean_spectrum:
        1D mean intensity vector aligned with ``raman_x``.
    raman_x:
        1D Raman-shift grid (cm⁻¹).
    anchor_cm1:
        Target anchors (cm⁻¹).
    half_width_cm1:
        Search half-width (cm⁻¹).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(detected_shifts, raw_peak_heights, baseline_subtracted_heights)``
        each with shape ``(n_anchors,)``, NaN where the window is empty.
    """

    y = np.asarray(mean_spectrum, dtype=float)
    x = np.asarray(raman_x, dtype=float)
    n = len(anchor_cm1)
    shifts = np.full(n, np.nan, dtype=float)
    raw_h = np.full(n, np.nan, dtype=float)
    adj_h = np.full(n, np.nan, dtype=float)
    if x.size == 0 or y.size != x.size:
        return shifts, raw_h, adj_h

    shifts[:], raw_h[:], adj_h[:] = detect_targeted_peaks_on_spectrum_row(
        y,
        x,
        anchor_cm1,
        half_width_cm1=half_width_cm1,
    )
    return shifts, raw_h, adj_h
