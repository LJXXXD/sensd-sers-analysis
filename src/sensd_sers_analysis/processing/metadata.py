"""
Metadata preprocessing for SERS DataFrames.

Adds derived columns: log_concentration, concentration_group. Normalizes date.
Works on wide or tidy format; handles concentration as scalar or list-per-row.
"""

import numpy as np
import pandas as pd

from sensd_sers_analysis.utils.natural_sort import natural_sort

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


def add_target_concentration_group(df: pd.DataFrame) -> pd.DataFrame:
    """Label each recorded initial target exactly, without rounding actual values.

    Missing targets remain Unknown. Concentration labels describe intended dose,
    never sample identity; Sample Type identifies rinsate controls.
    """
    out = df.copy()
    target = extract_scalar_concentration(
        out.get("target_concentration", pd.Series(np.nan, index=out.index)), out
    )
    labels = target.map(lambda value: f"{value:g} CFU" if pd.notna(value) else "Unknown")
    out["target_concentration_group"] = pd.Categorical(
        labels, categories=natural_sort(labels.unique().tolist()), ordered=True
    )
    return out


def add_concentration_group(df: pd.DataFrame) -> pd.DataFrame:
    """Provide the nominal-group compatibility key used by analysis consumers.

    ``concentration_group`` aliases the exact initial-target labels. It never
    bins measured concentration, including measured zeros in bacterial samples.
    """
    out = add_target_concentration_group(df)
    out["concentration_group"] = out["target_concentration_group"].copy()
    return out


def preprocess_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log_concentration, concentration_group, and normalize date.

    Serotype values (``serotype`` column) are case-insensitive: strings are
    stripped and uppercased so labels such as ``"st"`` and ``"ST"`` merge.

    Works on wide or tidy format. For wide DataFrames, concentration may be
    a list per row (one per signal); uses signal_index to pick the scalar.

    Concentration groups retain exact initial targets; missing targets are Unknown.
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
