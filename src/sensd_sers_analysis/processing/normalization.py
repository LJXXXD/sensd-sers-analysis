"""Explicit exposure-time scaling of measured spectral intensities."""

import numpy as np
import pandas as pd

from sensd_sers_analysis.config.preprocessing import INTEGRATION_REFERENCE_MS


def integration_times_ms(frame: pd.DataFrame) -> pd.Series:
    """Return exposure times as numeric milliseconds, preserving invalid values as NaN."""
    values = frame.get("integration_time_ms", pd.Series(index=frame.index, dtype=float))
    return pd.to_numeric(values, errors="coerce")


def normalize_integration_time(
    frame: pd.DataFrame, *, reference_ms: float = INTEGRATION_REFERENCE_MS
) -> pd.DataFrame:
    """Scale spectral columns to a common exposure without mutating input metadata.

    The transformation is I_reference = I_measured * reference_ms / exposure_ms.
    It assumes linear unsaturated response and input intensities not already
    exposure-normalized. Scan averages are not summed exposure counts. Missing,
    non-finite or nonpositive exposure times raise rather than mixing scales.
    """
    if not np.isfinite(reference_ms) or reference_ms <= 0:
        raise ValueError("Reference integration time must be finite and positive.")
    times = integration_times_ms(frame)
    invalid = ~np.isfinite(times) | times.le(0)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} spectra have missing or invalid integration time.")
    output = frame.copy()
    columns = [column for column in frame if column.startswith("rs_")]
    output[columns] = frame[columns].astype(float).mul(reference_ms / times, axis=0)
    return output
