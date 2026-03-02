"""
Dynamic peak extraction from SERS spectra.

Serotype-specific, background-exclusive pipeline: anchors and windows are
computed per serovar using only >0 CFU samples. 0 CFU (turkey rinsate matrix)
is excluded from learning to avoid noise-driven variance inflation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from sensd_sers_analysis.data import get_raman_shift, get_signals_matrix


# Concentration label that must be excluded from learning
ZERO_CFU_LABEL = "0 CFU"

# Preferred concentration groups for anchor discovery (highest signal first)
_HIGH_CONC_PREFERENCE = ["1000 CFU", "100 CFU", "10 CFU", "1 CFU"]

# Default serotype for 0 CFU rows (when extracting background)
_DEFAULT_SEROTYPE_PREFERENCE = ("ST", "SE")

# Outer boundary: baseline threshold (fraction of peak height) when searching
# left/right from first/last anchor
OUTER_BASELINE_FRAC = 0.05

# Anchor discovery: permissive thresholds so biological peaks aren't suppressed
# (baseline drift can skew max-min range; use 1% prominence, 15-index distance)
ANCHOR_PROMINENCE_FRAC = 0.01
ANCHOR_DISTANCE_INDICES = 15


@dataclass
class PeakWindowInfo:
    """Metadata for one extracted peak."""

    peak_name: str
    center: float  # Anchor position on mean spectrum (Raman shift)
    window_min: float
    window_max: float
    success_rate: float  # fraction of non-NaN


def _is_zero_cfu(
    df: pd.DataFrame, concentration_group_col: str = "concentration_group"
) -> pd.Series:
    """Boolean mask: rows with concentration_group == '0 CFU'."""
    if concentration_group_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[concentration_group_col].astype(str) == ZERO_CFU_LABEL


def _exclude_zero_cfu(
    df: pd.DataFrame, concentration_group_col: str = "concentration_group"
) -> pd.DataFrame:
    """Filter out 0 CFU rows (background-only; no pathogen peaks)."""
    if df.empty:
        return df
    mask = ~_is_zero_cfu(df, concentration_group_col)
    return df.loc[mask]


def _find_high_conc_subset(
    df: pd.DataFrame,
    concentration_group_col: str = "concentration_group",
    *,
    exclude_zero: bool = True,
) -> pd.DataFrame:
    """Filter to highest available concentration for strong signal."""
    if exclude_zero:
        df = _exclude_zero_cfu(df, concentration_group_col)
    if df.empty or concentration_group_col not in df.columns:
        return df.head(0)
    vals = df[concentration_group_col].astype(str)
    for pref in _HIGH_CONC_PREFERENCE:
        mask = vals == pref
        if mask.any():
            return df.loc[mask]
    return df.head(0)


def _smooth_spectrum(
    y: np.ndarray, window_length: int = 11, polyorder: int = 3
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to reduce noise spikes.

    Uses odd window_length; falls back to no smoothing if too few points.
    """
    if len(y) < window_length or window_length < 3:
        return y
    if window_length % 2 == 0:
        window_length -= 1
    polyorder = min(polyorder, window_length - 1)
    return scipy_signal.savgol_filter(
        np.asarray(y, dtype=float), window_length, polyorder
    )


def _find_peaks_on_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    n_peaks: int,
    *,
    smooth_before_peaks: bool = True,
    prominence_frac: float = ANCHOR_PROMINENCE_FRAC,
    min_distance: int = ANCHOR_DISTANCE_INDICES,
) -> np.ndarray:
    """
    Find top N most prominent peaks on a 1D spectrum.

    Uses scipy.signal.find_peaks with permissive prominence (1% of range) and
    distance (15 indices) so biological peaks aren't suppressed by baseline
    drift. Peaks are sorted by prominence descending; top n_peaks are returned.
    """
    if len(x) < 3 or len(y) < 3 or n_peaks < 1:
        return np.array([], dtype=int)
    valid = np.isfinite(y)
    if not valid.all():
        y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        y_clean = np.asarray(y, dtype=float)
    if smooth_before_peaks:
        y_clean = _smooth_spectrum(y_clean)
    peak_range = np.max(y_clean) - np.min(y_clean) if np.ptp(y_clean) > 0 else 1.0
    prominence = max(prominence_frac * peak_range, 1e-10)
    # Use permissive distance so adjacent peaks aren't suppressed
    distance = min(min_distance, max(1, len(x) // max(1, n_peaks * 3)))
    peaks, props = scipy_signal.find_peaks(
        y_clean, prominence=prominence, distance=distance
    )
    if len(peaks) == 0:
        return np.array([], dtype=int)
    prom = props.get("prominences", np.ones(len(peaks)))
    order = np.argsort(-prom)
    top = peaks[order[:n_peaks]]
    return np.sort(top)


def _find_outer_left(y: np.ndarray, peak_idx: int, baseline_threshold: float) -> int:
    """From peak_idx, search left until y <= baseline_threshold or hit start."""
    i = peak_idx
    while i > 0:
        i -= 1
        if y[i] <= baseline_threshold:
            return i
    return 0


def _find_outer_right(y: np.ndarray, peak_idx: int, baseline_threshold: float) -> int:
    """From peak_idx, search right until y <= baseline_threshold or hit end."""
    n = len(y)
    i = peak_idx
    while i < n - 1:
        i += 1
        if y[i] <= baseline_threshold:
            return i
    return n - 1


def _compute_peak_windows_for_serotype(
    df_sero: pd.DataFrame,
    x: np.ndarray,
    n_peaks: int,
    concentration_group_col: str,
    min_window_padding: float,
) -> tuple[list[PeakWindowInfo], np.ndarray]:
    """
    Compute anchors and windows using "absolute minimum between anchors" method.

    Inner boundaries: argmin of mean spectrum between adjacent anchors (true
    valley, robust to jagged slopes). Outer boundaries: search left/right until
    signal drops to baseline (≤5% of peak height). Uses only >0 CFU samples.
    Returns (list of PeakWindowInfo, mean_spectrum).
    """
    if df_sero.empty:
        return [], np.array([])

    high_conc = _find_high_conc_subset(
        df_sero, concentration_group_col, exclude_zero=True
    )
    if high_conc.empty:
        high_conc = _exclude_zero_cfu(df_sero, concentration_group_col)
    if high_conc.empty:
        return [], np.array([])

    high_signals = get_signals_matrix(high_conc)
    if high_signals.size == 0 or not np.any(np.isfinite(high_signals)):
        mean_spec = np.zeros_like(x, dtype=float)
    else:
        mean_spec = np.nanmean(high_signals, axis=0)
    mean_spec = np.nan_to_num(mean_spec, nan=0.0)

    peak_indices = _find_peaks_on_spectrum(x, mean_spec, n_peaks)
    if len(peak_indices) == 0:
        return [], mean_spec

    # Sort by Raman shift: A1, A2, ..., Ak
    peak_indices = np.sort(peak_indices)
    n_anchors = len(peak_indices)
    anchors_x = x[peak_indices]

    # Inner boundaries: absolute minimum between adjacent anchors
    # boundary[i] = argmin(mean_spec) in [Ai, Ai+1] → right for peak i, left for peak i+1
    left_indices = np.zeros(n_anchors, dtype=int)
    right_indices = np.zeros(n_anchors, dtype=int)

    if n_anchors == 1:
        left_indices[0] = 0
        right_indices[0] = len(x) - 1
    else:
        for i in range(n_anchors - 1):
            lo, hi = int(peak_indices[i]), int(peak_indices[i + 1])
            segment = mean_spec[lo : hi + 1]
            argmin_rel = int(np.argmin(segment))
            valley_idx = lo + argmin_rel
            right_indices[i] = valley_idx
            left_indices[i + 1] = valley_idx

        # Outer left: Peak 1 — search left from A1 until baseline
        baseline_left = max(
            OUTER_BASELINE_FRAC * float(mean_spec[peak_indices[0]]),
            0.0,
            1e-10,
        )
        left_indices[0] = _find_outer_left(
            mean_spec, int(peak_indices[0]), baseline_left
        )

        # Outer right: Peak k — search right from Ak until baseline
        baseline_right = max(
            OUTER_BASELINE_FRAC * float(mean_spec[peak_indices[-1]]),
            0.0,
            1e-10,
        )
        right_indices[-1] = _find_outer_right(
            mean_spec, int(peak_indices[-1]), baseline_right
        )

    window_mins = np.array([float(x[left_indices[i]]) for i in range(n_anchors)])
    window_maxs = np.array([float(x[right_indices[i]]) for i in range(n_anchors)])

    peak_columns = [f"Peak_{i + 1}_Height" for i in range(n_anchors)]
    info_list = [
        PeakWindowInfo(
            peak_name=peak_columns[a_idx],
            center=float(anchors_x[a_idx]),
            window_min=float(window_mins[a_idx]),
            window_max=float(window_maxs[a_idx]),
            success_rate=0.0,  # Filled during extraction
        )
        for a_idx in range(n_anchors)
    ]
    return info_list, mean_spec


def _pick_default_serotype(
    available: list[str], preference: tuple[str, ...] = _DEFAULT_SEROTYPE_PREFERENCE
) -> str | None:
    """Pick default serotype for 0 CFU rows (e.g. ST, then SE)."""
    for p in preference:
        if p in available:
            return p
    return available[0] if available else None


def extract_dynamic_peak_features(
    df_wide: pd.DataFrame,
    n_peaks: int = 6,
    *,
    n_peaks_by_serotype: dict[str, int] | None = None,
    concentration_group_col: str = "concentration_group",
    serotype_col: str = "serotype",
    min_window_padding: float = 15.0,
    noise_threshold_frac: float = 0.02,
) -> tuple[
    pd.DataFrame,
    dict[str, list[PeakWindowInfo]],
    dict[str, np.ndarray],
    str | None,
    np.ndarray,
]:
    """
    Extract Peak_1_Height, Peak_2_Height, ... using serotype-specific pipeline.

    Rule 1: Exclude 0 CFU from learning (anchor discovery and ±3σ voting).
    Rule 2: Compute anchors and windows per serotype from high-conc mean.
    Rule 3: For each row, use that row's serotype windows; 0 CFU uses default.

    Args:
        df_wide: Wide DataFrame with rs_*, concentration_group, serotype.
        n_peaks: Default number of peaks (used when n_peaks_by_serotype is None
            or does not contain a serotype).
        n_peaks_by_serotype: Optional dict mapping serotype -> n_peaks for
            per-serotype peak count (e.g. {"ST": 4, "SE": 3}).
        concentration_group_col: Column for concentration labels.
        serotype_col: Column for serotype/serovar.
        min_window_padding: Minimum half-width of window (cm⁻¹).
        noise_threshold_frac: Min prominence as fraction of max intensity.

    Returns:
        Tuple of (
            DataFrame with metadata+Peak_i_Height,
            dict[serotype -> list[PeakWindowInfo]],
            dict[serotype -> mean_spectrum],
            default_serotype (for 0 CFU),
            raman_shift array
        ).
    """
    if df_wide.empty or n_peaks < 1:
        return (
            pd.DataFrame(),
            {},
            {},
            None,
            np.array([]),
        )

    signals = get_signals_matrix(df_wide)
    raman_shift = get_raman_shift(df_wide)
    x = np.asarray(raman_shift, dtype=float)
    if x.size < 3:
        return df_wide[[]].copy(), {}, {}, None, x

    def _n_for_sero(sero: str) -> int:
        if n_peaks_by_serotype:
            return n_peaks_by_serotype.get(sero, n_peaks)
        return n_peaks

    if serotype_col not in df_wide.columns:
        # Fallback: single global (old behavior, but still exclude 0 CFU)
        df_pos = _exclude_zero_cfu(df_wide, concentration_group_col)
        if df_pos.empty:
            return df_wide[[]].copy(), {}, {}, None, x
        n_use = _n_for_sero("(all)")
        info_list, mean_spec = _compute_peak_windows_for_serotype(
            df_pos, x, n_use, concentration_group_col, min_window_padding
        )
        if not info_list:
            return df_wide[[]].copy(), {}, {}, None, x
        by_sero = {"(all)": info_list}
        mean_by_sero = {"(all)": mean_spec}
        default_sero = "(all)"
    else:
        by_sero: dict[str, list[PeakWindowInfo]] = {}
        mean_by_sero: dict[str, np.ndarray] = {}
        serotypes = df_wide[serotype_col].dropna().unique().astype(str).tolist()
        serotypes = [s for s in serotypes if s and s != "nan"]

        for sero in serotypes:
            df_sero = df_wide[df_wide[serotype_col].astype(str) == sero]
            n_use = _n_for_sero(sero)
            info_list, mean_spec = _compute_peak_windows_for_serotype(
                df_sero, x, n_use, concentration_group_col, min_window_padding
            )
            if info_list:
                by_sero[sero] = info_list
                mean_by_sero[sero] = mean_spec

        default_sero = _pick_default_serotype(list(by_sero.keys()))

    if not by_sero:
        return df_wide[[]].copy(), {}, {}, None, x

    # Max peaks across serotypes defines output columns (fewer → NaN)
    max_peaks = max(len(infos) for infos in by_sero.values())
    peak_columns = [f"Peak_{i + 1}_Height" for i in range(max_peaks)]
    n_anchors = max_peaks
    n_samples = signals.shape[0]

    global_max_int = np.nanmax(signals) if np.any(np.isfinite(signals)) else 1.0
    noise_threshold = noise_threshold_frac * global_max_int

    # Extraction: per row, use serotype-specific windows
    heights = np.full((n_samples, n_anchors), np.nan, dtype=float)
    zero_cfu_mask = _is_zero_cfu(df_wide, concentration_group_col).values

    for i in range(n_samples):
        y_row = np.nan_to_num(signals[i], nan=0.0)
        if zero_cfu_mask[i]:
            sero = default_sero
        elif serotype_col in df_wide.columns:
            sero = str(df_wide[serotype_col].iloc[i])
        else:
            sero = default_sero
        if sero not in by_sero:
            sero = default_sero
        if sero is None:
            continue
        infos = by_sero[sero]
        for a_idx, info in enumerate(infos):
            w_min, w_max = info.window_min, info.window_max
            mask = (x >= w_min) & (x <= w_max)
            if not mask.any():
                continue
            window_y = y_row[mask]
            n_edge = max(1, int(len(window_y) * 0.1))
            baseline = np.mean(np.concatenate([window_y[:n_edge], window_y[-n_edge:]]))
            peak_height = np.max(window_y) - baseline
            if peak_height >= noise_threshold:
                heights[i, a_idx] = peak_height

    # Update success_rate per serotype
    for sero, infos in by_sero.items():
        if serotype_col in df_wide.columns:
            sero_mask = (df_wide[serotype_col].astype(str) == sero).values
        else:
            sero_mask = np.ones(n_samples, dtype=bool)
        n_sero = int(np.sum(sero_mask))
        if n_sero > 0:
            for a_idx, info in enumerate(infos):
                count = int(np.sum(np.isfinite(heights[sero_mask, a_idx])))
                infos[a_idx] = PeakWindowInfo(
                    peak_name=info.peak_name,
                    center=info.center,
                    window_min=info.window_min,
                    window_max=info.window_max,
                    success_rate=float(count / n_sero),
                )

    metadata_cols = [
        c for c in df_wide.columns if not (isinstance(c, str) and c.startswith("rs_"))
    ]
    out = df_wide[metadata_cols].copy()
    for a_idx, col in enumerate(peak_columns):
        out[col] = heights[:, a_idx]

    return out, by_sero, mean_by_sero, default_sero, x


def get_peak_height_columns(peak_infos: list[PeakWindowInfo]) -> list[str]:
    """Return list of Peak_i_Height column names from PeakWindowInfo."""
    return [p.peak_name for p in peak_infos]
