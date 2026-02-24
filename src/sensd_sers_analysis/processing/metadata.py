"""
Metadata preprocessing for SERS DataFrames.

Adds derived columns: log_concentration, concentration_group. Normalizes date.
Works on wide or tidy format; handles concentration as scalar or list-per-row.
"""

import numpy as np
import pandas as pd

from sensd_sers_analysis.utils.natural_sort import natural_sort

# Log10 of group centers (1, 10, 100, 1000). 0 CFU has no log.
_CONC_GROUP_CENTERS_LOG = np.array([0.0, 1.0, 2.0, 3.0])  # log10(1), log10(10), ...
_CONC_GROUP_LABELS = ["1 CFU", "10 CFU", "100 CFU", "1000 CFU"]

# Ordered categories for pd.Categorical; pure text ("Unknown") sorts last
_CONC_CATEGORIES = natural_sort(
    ["0 CFU", "1 CFU", "10 CFU", "100 CFU", "1000 CFU", "Unknown"]
)


def _extract_scalar_concentration(series: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    Extract scalar concentration per row from concentration column.

    When concentration is a list (one per signal), uses signal_index to pick
    the correct value. Otherwise uses the value as-is.
    """
    conc_vals = []
    for i in range(len(series)):
        c = series.iloc[i]
        if isinstance(c, (list, tuple)) and len(c) > 0:
            si = df["signal_index"].iloc[i] if "signal_index" in df.columns else 0
            idx = int(si) if pd.notna(si) else 0
            c = c[min(idx, len(c) - 1)]
        conc_vals.append(c)
    return pd.Series(pd.to_numeric(conc_vals, errors="coerce"), index=series.index)


def add_log_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log10(concentration) column. Handles concentration = 0.

    For concentration > 0: log_concentration = log10(concentration).
    For concentration <= 0 or NaN: log_concentration = NaN (log(0) undefined).

    Args:
        df: DataFrame with concentration column (scalar or list per row).

    Returns:
        Copy of df with log_concentration column added.
    """
    out = df.copy()
    if "concentration" not in out.columns:
        return out

    conc = _extract_scalar_concentration(out["concentration"], out)
    log_conc = np.where(
        conc.notna() & (conc > 0),
        np.log10(conc),
        np.nan,
    )
    out["log_concentration"] = pd.Series(log_conc, index=out.index, dtype=float)
    return out


def add_concentration_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add concentration_group: assign each signal to nearest log10 center.

    Group centers: 1, 10, 100, 1000 CFU (log10: 0, 1, 2, 3).
    Concentration 0 is assigned "0 CFU" (no log exists).
    For conc > 0: compute log10(conc) and assign to nearest center.

    Args:
        df: DataFrame with concentration column (scalar or list per row).

    Returns:
        Copy of df with concentration_group column (ordered Categorical).
    """
    out = df.copy()
    cat_dtype = pd.CategoricalDtype(categories=_CONC_CATEGORIES, ordered=True)
    out["concentration_group"] = pd.Categorical(
        ["Unknown"] * len(out),
        categories=_CONC_CATEGORIES,
        ordered=True,
    )

    if "concentration" not in out.columns:
        return out

    conc = _extract_scalar_concentration(out["concentration"], out)
    valid = conc.notna()

    # Concentration 0 -> "0 CFU"
    zero_mask = valid & (conc <= 0)
    out.loc[zero_mask, "concentration_group"] = "0 CFU"

    # Concentration > 0 -> nearest log10 center
    pos_mask = valid & (conc > 0)
    if pos_mask.any():
        log_conc = np.log10(conc[pos_mask].values)
        # Nearest of 0, 1, 2, 3 (log10 of 1, 10, 100, 1000)
        dists = np.abs(log_conc[:, np.newaxis] - _CONC_GROUP_CENTERS_LOG)
        nearest_idx = np.argmin(dists, axis=1)
        labels = [_CONC_GROUP_LABELS[i] for i in nearest_idx]
        out.loc[pos_mask, "concentration_group"] = pd.Categorical(
            labels, dtype=cat_dtype
        )

    return out


def preprocess_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log_concentration, concentration_group, and normalize date.

    Works on wide or tidy format. For wide DataFrames, concentration may be
    a list per row (one per signal); uses signal_index to pick the scalar.

    Binning: concentration 0 -> "0 CFU". conc > 0 -> nearest of
    log10(1), log10(10), log10(100), log10(1000).
    Date is normalized to YYYY-MM-DD string format.

    Args:
        df: DataFrame with metadata (and optionally concentration).

    Returns:
        Copy of df with added columns.
    """
    out = df.copy()
    out = add_log_concentration(out)
    out = add_concentration_group(out)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["date"] = out["date"].dt.strftime("%Y-%m-%d").fillna("").astype(str)

    return out
