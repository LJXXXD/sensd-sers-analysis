"""
SERS Data I/O for self-contained Excel files with embedded metadata.

Each file contains metadata (Sensor ID, Test ID, Connection ID, Serotype),
a concentration row, and Raman shift / intensity columns. Returns wide-format
DataFrames: one row per sample, metadata plus rs_* intensity columns. Use
get_signals_matrix, get_raman_shift, get_metadata_columns for ML pipelines;
wide_to_tidy for plotting.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_METADATA_KEYS = {"sensor id", "test id", "connection id", "serotype"}
CONCENTRATION_KEY_PATTERN = "concentration"
DEFAULT_PATTERN = "*.xlsx"

# Column names for sample identification and metadata
META_COLS = [
    "sensor_id",
    "test_id",
    "connection_id",
    "serotype",
    "concentration",
    "filename",
    "signal_index",
]
RAMAN_SHIFT_DECIMALS = 2
RS_COL_PREFIX = "rs_"


def _parse_embedded_format(
    file_path: Path,
) -> tuple[dict[str, str], np.ndarray, np.ndarray, list[float]]:
    """
    Parse a SERS Excel file in the embedded-metadata format.

    Returns:
        Tuple of (metadata_dict with normalized keys, raman_shift, signals_matrix,
        concentrations).
    """
    df = pd.read_excel(file_path, header=None)

    # Find concentration row: first row where col 0 contains "concentration"
    keys_norm = df[0].astype(str).str.strip().str.lower()
    conc_mask = keys_norm.str.contains(CONCENTRATION_KEY_PATTERN, na=False)
    if not conc_mask.any():
        raise ValueError(f"Concentration row not found in {file_path.name}")
    concentration_row_idx = int(conc_mask.idxmax())

    # Metadata block: rows before concentration row, key-value pairs
    meta_df = df.iloc[:concentration_row_idx, [0, 1]].dropna(how="any")
    meta_df[0] = meta_df[0].astype(str).str.strip().str.lower()
    meta_df[1] = meta_df[1].astype(str).str.strip()
    metadata = dict(zip(meta_df[0], meta_df[1]))

    # Concentrations: use valid column indices to avoid misalignment when blanks exist
    conc_row = df.iloc[concentration_row_idx, 1:]
    conc_numeric = pd.to_numeric(conc_row, errors="coerce")
    valid_mask = conc_numeric.notna()
    valid_col_indices = conc_numeric.index[valid_mask].tolist()
    concentrations = conc_numeric.loc[valid_col_indices].tolist()
    if not concentrations:
        raise ValueError(
            f"No valid concentrations in row {concentration_row_idx + 1} of {file_path.name}"
        )

    # Data block: locate by first row where col 0 is numeric (Raman shift)
    after_conc = df.iloc[concentration_row_idx + 1 :]
    raman_col = after_conc[0]
    numeric_mask = pd.to_numeric(raman_col, errors="coerce").notna()
    first_data_idx = int(numeric_mask.idxmax()) if numeric_mask.any() else None
    if first_data_idx is None:
        raise ValueError(f"No signal data found in {file_path.name}")

    # Use exact valid_col_indices for signal columns (col 0 = raman)
    data_df = df.loc[first_data_idx:, [0] + valid_col_indices].copy()
    data_valid = data_df.dropna(subset=[0])
    raman_shift = pd.to_numeric(data_valid[0], errors="coerce").values
    signal_cols = valid_col_indices
    signals = data_valid[signal_cols].astype(float).values

    if np.isnan(signals).any():
        raise ValueError(f"NaN detected in signals in {file_path.name}")

    missing = REQUIRED_METADATA_KEYS - metadata.keys()
    if missing:
        raise ValueError(
            f"Required metadata {sorted(missing)} not found in {file_path.name}"
        )

    return metadata, raman_shift, signals, concentrations


def _load_signal_file(file_path: str | Path) -> pd.DataFrame:
    """Load one SERS Excel file; returns wide-format DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    metadata, raman_shift, signals, concentrations = _parse_embedded_format(path)
    n_signals = signals.shape[1]
    rs_rounded = np.round(raman_shift, RAMAN_SHIFT_DECIMALS)
    rs_col_names = [f"{RS_COL_PREFIX}{v:.{RAMAN_SHIFT_DECIMALS}f}" for v in rs_rounded]

    meta_df = pd.DataFrame(
        {
            "sensor_id": metadata.get("sensor id", ""),
            "test_id": metadata.get("test id", ""),
            "connection_id": metadata.get("connection id", ""),
            "serotype": metadata.get("serotype", ""),
            "concentration": concentrations,
            "filename": path.name,
            "signal_index": np.arange(n_signals),
        }
    )
    signals_df = pd.DataFrame(signals.T, columns=rs_col_names)
    return pd.concat([meta_df, signals_df], axis=1)


def _collect_files(
    paths: Union[str, Path, List[Union[str, Path]]], pattern: str
) -> List[Path]:
    """Resolve paths to a flat list of Excel files. Handles file/folder or mix."""
    if not isinstance(paths, list):
        paths = [paths]
    files: List[Path] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            logger.warning("Path does not exist, skipping: %s", path)
            continue
        if path.is_file():
            if path.suffix.lower() in (".xlsx", ".xls"):
                files.append(path)
            else:
                logger.warning("Skipping non-Excel file: %s", path)
        else:
            for f in path.glob(pattern):
                if f.is_file() and not f.name.startswith(("~", "_")):
                    files.append(f)
    return files


def load_sers_data(
    paths: Union[str, Path, List[Union[str, Path]]],
    *,
    serotypes: Optional[List[str]] = None,
    pattern: str = DEFAULT_PATTERN,
) -> pd.DataFrame:
    """
    Load SERS data from file(s) and/or folder(s).

    Args:
        paths: A file path, folder path, or list of files and/or folders.
        serotypes: If provided, only load files whose Serotype metadata matches.
        pattern: Glob pattern when scanning folders (default: *.xlsx).

    Returns:
        Wide DataFrame with META_COLS + rs_* intensity columns (one row per sample).
    """
    files = _collect_files(paths, pattern)
    if not files:
        return pd.DataFrame()

    dfs: List[pd.DataFrame] = []
    for f in files:
        try:
            df = _load_signal_file(f)
            if df.empty:
                continue
            if serotypes is not None and df["serotype"].iloc[0] not in serotypes:
                continue
            dfs.append(df)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipping file %s: %s", f.name, e)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _get_raman_columns(df: pd.DataFrame) -> list:
    """Return sorted list of Raman shift column names (rs_* cols)."""
    rs_cols = [
        c for c in df.columns if isinstance(c, str) and c.startswith(RS_COL_PREFIX)
    ]
    return sorted(rs_cols, key=lambda c: float(c[len(RS_COL_PREFIX) :]))


def get_signals_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extract (n_samples, n_features) array from wide DataFrame for ML.

    Returns:
        Intensity matrix; rows = samples, columns = Raman shifts (sorted).
    """
    rs_cols = _get_raman_columns(df)
    if not rs_cols:
        raise ValueError(
            "DataFrame has no Raman intensity columns; expected wide format"
        )
    return df[rs_cols].values


def get_raman_shift(df: pd.DataFrame) -> np.ndarray:
    """Return the Raman shift (wavenumber) array for the spectral grid."""
    rs_cols = _get_raman_columns(df)
    if not rs_cols:
        raise ValueError(
            "DataFrame has no Raman intensity columns; expected wide format"
        )
    return np.array([float(c[len(RS_COL_PREFIX) :]) for c in rs_cols])


def get_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return metadata (sensor_id, serotype, concentration, etc.) as a DataFrame."""
    available = [c for c in META_COLS if c in df.columns]
    if not available:
        return pd.DataFrame()
    return df[available].reset_index(drop=True)


def wide_to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide DataFrame to long format (raman_shift, intensity) for plotting.
    """
    rs_cols = _get_raman_columns(df)
    id_cols = [c for c in META_COLS if c in df.columns]
    tidy = df.melt(
        id_vars=id_cols,
        value_vars=rs_cols,
        var_name="_rs_col",
        value_name="intensity",
    )
    tidy["raman_shift"] = tidy["_rs_col"].str[len(RS_COL_PREFIX) :].astype(float)
    return tidy.drop(columns=["_rs_col"])
