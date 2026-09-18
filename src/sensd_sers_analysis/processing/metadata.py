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
_CONC_CATEGORIES = natural_sort(["0 CFU", "1 CFU", "10 CFU", "100 CFU", "1000 CFU", "Unknown"])

_INVALID_SEROTYPE_STRINGS = frozenset(("", "NAN", "NONE"))


def sorted_unique_canonical_serotypes(df: pd.DataFrame, *, column: str = "serotype") -> list[str]:
    """
    Return sorted unique serotype labels with case-insensitive de-duplication.

    Labels are normalized to stripped uppercase ASCII (e.g. ``"st"`` and
    ``"ST"`` both become ``"ST"``).

    Parameters
    ----------
    df:
        Dataframe that may contain a serotype column.
    column:
        Metadata column name.

    Returns
    -------
    list[str]
        Sorted canonical labels; empty when the column is missing or all invalid.
    """

    if df is None or df.empty or column not in df.columns:
        return []
    sero = df[column].dropna()
    if sero.empty:
        return []
    normed = sero.astype(str).str.strip().str.upper()
    normed = normed.mask(normed.isin(_INVALID_SEROTYPE_STRINGS), pd.NA).dropna()
    if normed.empty:
        return []
    return sorted(pd.unique(normed).tolist())


def _normalize_serotype_column_inplace(out: pd.DataFrame) -> None:
    """
    Canonicalize ``serotype`` strings in-place (strip + uppercase).

    Empty or placeholder strings become missing.

    Parameters
    ----------
    out:
        Dataframe copy that may contain ``serotype``.
    """

    if "serotype" not in out.columns:
        return
    mask = out["serotype"].notna()
    if not mask.any():
        return
    coerced = out.loc[mask, "serotype"].astype(str).str.strip().str.upper()
    coerced = coerced.mask(coerced.isin(_INVALID_SEROTYPE_STRINGS), pd.NA)
    out.loc[mask, "serotype"] = coerced


def extract_scalar_concentration(series: pd.Series, df: pd.DataFrame) -> pd.Series:
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


# Backwards-compatible alias for existing internal imports.
_extract_scalar_concentration = extract_scalar_concentration


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

    conc = extract_scalar_concentration(out["concentration"], out)
    log_conc = np.full(len(conc), np.nan, dtype=float)
    pos_mask = conc.notna() & (conc > 0)
    log_conc[pos_mask] = np.log10(conc[pos_mask].astype(float))
    out["log_concentration"] = pd.Series(log_conc, index=out.index, dtype=float)
    return out


def _bin_concentration_group(conc: pd.Series) -> pd.Categorical:
    """
    Bin scalar CFU/mL values to ordered concentration-group labels.

    Group centers: 1, 10, 100, 1000 CFU (log10: 0, 1, 2, 3). Concentration 0
    (or negative) maps to ``"0 CFU"``; positive values snap to the nearest
    log10 center; missing values map to ``"Unknown"``.

    Args:
        conc: Numeric CFU/mL per row (NaN allowed).

    Returns:
        Ordered ``pd.Categorical`` of group labels aligned to ``conc``.
    """
    cat_dtype = pd.CategoricalDtype(categories=_CONC_CATEGORIES, ordered=True)
    labels = pd.Series(["Unknown"] * len(conc), index=conc.index, dtype=object)

    valid = conc.notna()
    zero_mask = valid & (conc <= 0)
    labels.loc[zero_mask] = "0 CFU"

    pos_mask = valid & (conc > 0)
    if pos_mask.any():
        log_conc = np.log10(conc[pos_mask].astype(float).values)
        # Nearest of 0, 1, 2, 3 (log10 of 1, 10, 100, 1000)
        dists = np.abs(log_conc[:, np.newaxis] - _CONC_GROUP_CENTERS_LOG)
        nearest_idx = np.argmin(dists, axis=1)
        labels.loc[pos_mask] = [_CONC_GROUP_LABELS[i] for i in nearest_idx]

    return pd.Categorical(labels, dtype=cat_dtype)


def add_concentration_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add concentration_group from the **actual** concentration.

    Assigns each signal to the nearest log10 center of its measured CFU/mL.
    This column reflects the actual (plate-count) concentration, so downstream
    consumers can quantify sample-to-sample variability. Use
    :func:`add_target_concentration_group` for grouping by intended dose.

    Args:
        df: DataFrame with concentration column (scalar or list per row).

    Returns:
        Copy of df with concentration_group column (ordered Categorical).
    """
    out = df.copy()
    if "concentration" not in out.columns:
        out["concentration_group"] = pd.Categorical(
            ["Unknown"] * len(out),
            categories=_CONC_CATEGORIES,
            ordered=True,
        )
        return out

    conc = extract_scalar_concentration(out["concentration"], out)
    out["concentration_group"] = _bin_concentration_group(conc)
    return out


def add_target_concentration_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add target_concentration_group from the **nominal target** concentration.

    Bins the intended dose (``target_concentration``) into 0/1/10/100/1000 CFU
    groups. Missing initial targets remain Unknown for explicit sample-type
    datasets. Only legacy dataframes without sample_type fall back to the
    actual-derived ``concentration_group``. This is the canonical grouping key for QC drill-down and the
    validation summary tables, while sample-to-sample variability is measured
    from the actual ``concentration``.

    Args:
        df: DataFrame with target_concentration and/or concentration columns.

    Returns:
        Copy of df with target_concentration_group column (ordered Categorical).
    """
    out = df.copy()
    if "target_concentration" not in out.columns:
        # No target available: mirror the actual-derived group when present.
        if "concentration_group" in out.columns:
            out["target_concentration_group"] = pd.Categorical(
                out["concentration_group"].astype(str),
                categories=_CONC_CATEGORIES,
                ordered=True,
            )
        else:
            out["target_concentration_group"] = pd.Categorical(
                ["Unknown"] * len(out),
                categories=_CONC_CATEGORIES,
                ordered=True,
            )
        return out

    target = extract_scalar_concentration(out["target_concentration"], out)
    target_group = _bin_concentration_group(target)

    # Fall back to the actual-derived group where the target is missing.
    if "concentration_group" in out.columns and "sample_type" not in out.columns:
        target_labels = pd.Series(
            np.asarray(target_group.astype(str)), index=out.index, dtype=object
        )
        missing_target = target.isna()
        target_labels.loc[missing_target] = (
            out.loc[missing_target, "concentration_group"].astype(str).values
        )
        cat_dtype = pd.CategoricalDtype(categories=_CONC_CATEGORIES, ordered=True)
        out["target_concentration_group"] = pd.Categorical(target_labels, dtype=cat_dtype)
    else:
        out["target_concentration_group"] = target_group

    return out


def preprocess_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log_concentration, concentration_group, and normalize date.

    Serotype values (``serotype`` column) are case-insensitive: strings are
    stripped and uppercased so labels such as ``"st"`` and ``"ST"`` merge.

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
    _normalize_serotype_column_inplace(out)
    if "special_treatment" in out.columns:
        out["special_treatment"] = (
            out["special_treatment"].fillna("").astype(str).str.strip().replace("", "None")
        )
    out = add_log_concentration(out)
    out = add_concentration_group(out)
    out = add_target_concentration_group(out)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", format="mixed")
        out["date"] = out["date"].dt.strftime("%Y-%m-%d").fillna("").astype(str)

    return out


def sample_type_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return explicit control and bacteria masks; unknown identities match neither.

    Concentration is a measurement, not sample identity. Missing Sample Type
    requires source metadata completion before classification or control analysis.
    """
    values = df.get("sample_type", pd.Series("", index=df.index))
    values = values.fillna("").astype(str).str.strip().str.casefold()
    return values.eq("rinsate control"), values.eq("bacteria sample")
