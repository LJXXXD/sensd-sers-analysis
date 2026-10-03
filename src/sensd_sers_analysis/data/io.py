"""
SERS Data I/O for self-contained Excel files with embedded metadata.

Each file contains metadata (Sensor ID, Test ID, Connection ID, Serotype),
a concentration row, and Raman shift / intensity columns. Returns wide-format
DataFrames: one row per sample, metadata plus rs_* intensity columns. Use
get_signals_matrix, get_raman_shift, get_metadata_columns for ML pipelines;
wide_to_tidy for plotting.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SersLoadReport:
    """
    Per-file outcomes from a batch SERS Excel load.

    Parameters
    ----------
    loaded_files:
        Basenames of workbooks parsed successfully.
    skipped_files:
        ``(filename, user_message)`` pairs for workbooks that could not be loaded.
    """

    loaded_files: tuple[str, ...] = ()
    skipped_files: tuple[tuple[str, str], ...] = ()

    @property
    def n_loaded(self) -> int:
        """Return the number of successfully loaded files."""
        return len(self.loaded_files)

    @property
    def n_skipped(self) -> int:
        """Return the number of skipped files."""
        return len(self.skipped_files)


REQUIRED_METADATA_KEYS = {"sensor id", "test id", "connection id", "serotype"}
CONCENTRATION_KEY_PATTERN = "concentration"
DEFAULT_PATTERN = "*.xlsx"
RAMAN_SHIFT_ROW_LABEL = "raman shift"

# Excel column-A labels for optional per-signal rows (after concentration rows).
PER_SIGNAL_ROW_LABEL_TO_COLUMN = {
    "file name": "source_txt_filename",
    "special treatment": "special_treatment",
    "sample type": "sample_type",
}

# Preferred display/order for known metadata columns. The loader is generic and
# will surface ANY file-level metadata row as a column (see
# ``_normalize_metadata_key``); this list only controls ordering for the known
# fields. Unknown/new rows are appended after these, so adding metadata rows to
# the Excel template stays forward-compatible without touching the loader.
META_COLS = [
    # Instrument geometry
    "disk_diameter_nm",
    "periodicity_um",
    "thickness_nm",
    "core_diameter_um",
    "sensor_model",
    # Acquisition settings
    "integration_time_ms",
    "scan_average",
    # Sample / run identification
    "sensor_id",
    "test_id",
    "connection_id",
    "serotype",
    "rinsate_type",
    "date",
    "testing_time",
    "operator",
    "notes",
    # Per-signal columns
    "target_concentration",
    "concentration",
    "filename",
    "source_txt_filename",
    "special_treatment",
    "sample_type",
    "signal_index",
]

# Per-signal columns produced from the concentration block and provenance rows.
# These must not be overwritten by generic file-level metadata parsing.
PER_SIGNAL_COLUMNS = frozenset(
    {
        "target_concentration",
        "concentration",
        "filename",
        "source_txt_filename",
        "special_treatment",
        "sample_type",
        "signal_index",
    }
)

# Known file-level metadata columns guaranteed to exist on every loaded file
# (blank when the source row is empty). Keeps the schema stable across files so
# optional fields like ``notes`` or ``testing_time`` do not silently disappear
# when one workbook leaves them blank.
KNOWN_FILE_LEVEL_COLUMNS = tuple(c for c in META_COLS if c not in PER_SIGNAL_COLUMNS)
RAMAN_SHIFT_DECIMALS = 2
RS_COL_PREFIX = "rs_"

# Computed fields cannot be supplied as file-level metadata and overwritten later.
RESERVED_METADATA_COLUMNS = frozenset(
    {
        "raman_shift",
        "intensity",
        "log_concentration",
        "concentration_group",
        "target_concentration_group",
        "target",
        "max_intensity",
        "mean_intensity",
        "integral_area",
    }
)


def _normalize_metadata_key(raw_key: str) -> str:
    """
    Convert a raw column-A metadata label into a canonical snake_case column name.

    The mapping mirrors the prep tool's field keys (e.g. ``"Disk Diameter (nm)"``
    -> ``"disk_diameter_nm"``, ``"Core Diameter (µm)"`` -> ``"core_diameter_um"``)
    so units are preserved in the name and micro signs collapse to ``u``. Any
    label the loader has never seen is still converted deterministically, which
    keeps the loader forward-compatible with new metadata rows.

    Parameters
    ----------
    raw_key:
        Raw label read from column A of the workbook.

    Returns
    -------
    str
        Snake_case column name (empty string if the label has no usable chars).
    """

    key = str(raw_key).strip().lower()
    key = key.replace("µ", "u").replace("μ", "u")
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def _normalize_column_a_labels(series: pd.Series) -> pd.Series:
    """
    Normalize column-A metadata labels for reliable string matching.

    Blank Excel cells are read as ``NaN``; without ``fillna`` they remain float
    and break substring checks such as ``"concentration" in label``.
    """

    return series.fillna("").astype(str).str.strip().str.lower()


def _user_facing_load_error(filename: str, exc: BaseException) -> str:
    """
    Convert a parser exception into a short, user-readable load message.

    Parameters
    ----------
    filename:
        Basename of the workbook being loaded.
    exc:
        Exception raised while parsing the file.

    Returns
    -------
    str
        Message suitable for display in the Streamlit sidebar.
    """

    if isinstance(exc, ValueError):
        message = str(exc)
        if filename in message:
            return message
        return f"{filename}: {message}"
    if isinstance(exc, FileNotFoundError):
        return f"{filename}: file not found."
    return (
        f"{filename}: could not read this workbook ({type(exc).__name__}). "
        "If the file was edited outside **Convert TXT → Excel**, "
        "re-export it or verify that column A metadata labels are present."
    )


def _parse_per_signal_label_rows(
    df: pd.DataFrame,
    *,
    valid_col_indices: list,
    first_concentration_row_idx: int,
    concentration_row_idx: int,
) -> dict[str, list[str]]:
    """
    Read per-signal label rows above or below the concentration block.

    Supports the canonical order (File Name, Special Treatment, then
    concentrations) and legacy workbooks that placed label rows after Actual
    Concentration.

    Parameters
    ----------
    df:
        Full worksheet as read with ``header=None``.
    valid_col_indices:
        Column indices with valid numeric concentrations.
    first_concentration_row_idx:
        Row index of the first concentration-labeled row.
    concentration_row_idx:
        Row index used for numeric concentration alignment (Actual when present).

    Returns
    -------
    dict[str, list[str]]
        Mapping of dataframe column name to one label per signal column.
    """

    keys_norm = _normalize_column_a_labels(df[0])
    parsed: dict[str, list[str]] = {}

    def _read_row(row_idx: int) -> None:
        label = keys_norm.iloc[row_idx]
        if not label:
            return
        column_name = PER_SIGNAL_ROW_LABEL_TO_COLUMN.get(label)
        if column_name is None:
            return
        values: list[str] = []
        for col_idx in valid_col_indices:
            cell = df.iloc[row_idx, col_idx]
            values.append("" if pd.isna(cell) else str(cell).strip())
        parsed[column_name] = values

    for row_idx in range(0, first_concentration_row_idx):
        _read_row(row_idx)

    for row_idx in range(concentration_row_idx + 1, len(df)):
        label = keys_norm.iloc[row_idx]
        if label == RAMAN_SHIFT_ROW_LABEL:
            break
        _read_row(row_idx)

    return parsed


def _parse_file_metadata_block(
    df: pd.DataFrame,
    *,
    first_concentration_row_idx: int,
) -> dict[str, str]:
    """
    Parse file-level metadata key-value pairs before the concentration block.

    Skips per-signal label rows (File Name, Special Treatment) that may appear
    immediately above Target Concentration.
    """

    metadata: dict[str, str] = {}
    keys_norm = _normalize_column_a_labels(df[0])
    for row_idx in range(first_concentration_row_idx):
        label = keys_norm.iloc[row_idx]
        if not label:
            continue
        if label in PER_SIGNAL_ROW_LABEL_TO_COLUMN:
            continue
        if CONCENTRATION_KEY_PATTERN in label:
            continue
        key_cell = df.iloc[row_idx, 0]
        value_cell = df.iloc[row_idx, 1]
        if pd.isna(key_cell) or pd.isna(value_cell):
            continue
        key = str(key_cell).strip().lower()
        value = str(value_cell).strip()
        if key:
            if key in metadata:
                raise ValueError(f"Duplicate file-level metadata field: {key!r}")
            metadata[key] = value
    return metadata


def _align_per_signal_values(
    values: list[str] | None,
    *,
    n_signals: int,
) -> list[str]:
    """Pad or trim per-signal string lists to ``n_signals`` length."""

    if not values:
        return [""] * n_signals
    if len(values) >= n_signals:
        return values[:n_signals]
    return values + [""] * (n_signals - len(values))


def _parse_embedded_format(
    file_path: Path,
) -> tuple[dict[str, str], np.ndarray, np.ndarray, list[float], list[float], dict[str, list[str]]]:
    """
    Parse a SERS Excel file in the embedded-metadata format.

    Returns:
        Tuple of (metadata_dict with normalized keys, raman_shift, signals_matrix,
        target_concentrations, actual_concentrations, per_signal_labels).

    Notes
    -----
    Numeric analysis uses the **Actual Concentration** row when present (the
    measured CFU/mL per signal). The **Target Concentration** row (the nominal
    dosing level) is captured separately so downstream grouping can rely on the
    intended target while sample-to-sample variability is derived from the
    actual values. Legacy single-row workbooks expose only actual values; target
    values are then filled with ``NaN``.
    """
    df = pd.read_excel(file_path, header=None)

    # Find concentration row: first row where col 0 contains "concentration"
    keys_norm = _normalize_column_a_labels(df[0])
    conc_mask = keys_norm.str.contains(CONCENTRATION_KEY_PATTERN, na=False)
    if not conc_mask.any():
        raise ValueError(f"Concentration row not found in {file_path.name}")
    first_concentration_row_idx = int(conc_mask.idxmax())

    actual_mask = keys_norm.str.contains("actual concentration", na=False)
    if actual_mask.any():
        concentration_row_idx = int(actual_mask.idxmax())
    else:
        concentration_row_idx = first_concentration_row_idx

    target_mask = keys_norm.str.contains("target concentration", na=False)
    if target_mask.any():
        target_row_idx: int | None = int(target_mask.idxmax())
    elif first_concentration_row_idx != concentration_row_idx:
        target_row_idx = first_concentration_row_idx
    else:
        target_row_idx = None

    metadata = _parse_file_metadata_block(
        df,
        first_concentration_row_idx=first_concentration_row_idx,
    )

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

    # Missing targets remain unavailable; actual values never supply nominal labels.
    if target_row_idx is not None:
        target_numeric = pd.to_numeric(df.iloc[target_row_idx, 1:], errors="coerce")
        target_concentrations = target_numeric.loc[valid_col_indices].tolist()
    else:
        target_concentrations = [float("nan")] * len(valid_col_indices)

    # Data block: locate by first row where col 0 is numeric (Raman shift)
    after_conc = df.iloc[concentration_row_idx + 1 :]
    raman_col = after_conc[0]
    numeric_mask = pd.to_numeric(raman_col, errors="coerce").notna()
    first_data_idx = int(numeric_mask.idxmax()) if numeric_mask.any() else None
    if first_data_idx is None:
        raise ValueError(f"No signal data found in {file_path.name}")

    # Use exact valid_col_indices for signal columns (col 0 = raman)
    data_df = df.loc[first_data_idx:, [0] + valid_col_indices].copy()
    missing_coordinate = data_df[0].isna() & data_df[valid_col_indices].notna().any(axis=1)
    if missing_coordinate.any():
        raise ValueError(f"Missing Raman shift for rows containing signal data in {file_path.name}")
    data_valid = data_df.dropna(subset=[0])
    raman_shift = pd.to_numeric(data_valid[0], errors="coerce").values
    signal_cols = valid_col_indices
    signals = data_valid[signal_cols].astype(float).values

    if not np.isfinite(raman_shift).all():
        raise ValueError(f"Raman shifts must be finite in {file_path.name}")
    if not np.isfinite(signals).all():
        raise ValueError(f"Non-finite values detected in signals in {file_path.name}")

    missing = REQUIRED_METADATA_KEYS - metadata.keys()
    if missing:
        raise ValueError(f"Required metadata {sorted(missing)} not found in {file_path.name}")

    per_signal_labels = _parse_per_signal_label_rows(
        df,
        valid_col_indices=valid_col_indices,
        first_concentration_row_idx=first_concentration_row_idx,
        concentration_row_idx=concentration_row_idx,
    )

    return metadata, raman_shift, signals, target_concentrations, concentrations, per_signal_labels


def _load_signal_file(file_path: str | Path) -> pd.DataFrame:
    """Load one SERS Excel file; returns wide-format DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    metadata, raman_shift, signals, target_concentrations, concentrations, per_signal_labels = (
        _parse_embedded_format(path)
    )
    n_signals = signals.shape[1]
    rs_rounded = np.round(raman_shift, RAMAN_SHIFT_DECIMALS)
    if np.unique(rs_rounded).size != rs_rounded.size:
        raise ValueError(
            f"Raman shifts collide at {RAMAN_SHIFT_DECIMALS}-decimal precision in {path.name}"
        )
    rs_col_names = [f"{RS_COL_PREFIX}{v:.{RAMAN_SHIFT_DECIMALS}f}" for v in rs_rounded]

    source_txt_filenames = _align_per_signal_values(
        per_signal_labels.get("source_txt_filename"),
        n_signals=n_signals,
    )
    special_treatments = _align_per_signal_values(
        per_signal_labels.get("special_treatment"),
        n_signals=n_signals,
    )

    # Generic file-level metadata: every parsed row becomes a column so no
    # metadata is silently dropped. Known fields land on canonical names via
    # ``_normalize_metadata_key``; unknown fields are preserved verbatim.
    file_level_columns: dict[str, object] = {}
    for raw_key, value in metadata.items():
        column = _normalize_metadata_key(raw_key)
        if (
            not column
            or column in PER_SIGNAL_COLUMNS | RESERVED_METADATA_COLUMNS
            or column.startswith((RS_COL_PREFIX, "peak_near_"))
        ):
            raise ValueError(f"Reserved or unusable file-level metadata field: {raw_key!r}")
        if column in file_level_columns:
            raise ValueError(f"Duplicate normalized metadata field: {column!r}")
        file_level_columns[column] = value

    # Guarantee a stable schema: known file-level fields always exist (blank
    # when the workbook left them empty) so downstream code can rely on them.
    for column in KNOWN_FILE_LEVEL_COLUMNS:
        file_level_columns.setdefault(column, "")

    per_signal_columns: dict[str, object] = {
        "target_concentration": target_concentrations,
        "concentration": concentrations,
        "filename": path.name,
        "source_txt_filename": source_txt_filenames,
        "special_treatment": special_treatments,
        "sample_type": _align_per_signal_values(
            per_signal_labels.get("sample_type"), n_signals=n_signals
        ),
        "signal_index": np.arange(n_signals),
    }

    meta_df = pd.DataFrame({**file_level_columns, **per_signal_columns})
    ordered = [c for c in META_COLS if c in meta_df.columns]
    extras = [c for c in meta_df.columns if c not in META_COLS]
    meta_df = meta_df[ordered + extras]

    signals_df = pd.DataFrame(signals.T, columns=rs_col_names)
    return pd.concat([meta_df, signals_df], axis=1)


def _collect_files(paths: Union[str, Path, List[Union[str, Path]]], pattern: str) -> List[Path]:
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


def _load_sers_data_batch(
    files: List[Path],
    *,
    serotypes: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, SersLoadReport]:
    """
    Load a flat list of Excel workbooks into one wide dataframe.

    Parameters
    ----------
    files:
        Resolved workbook paths.
    serotypes:
        If provided, only retain files whose Serotype metadata matches.

    Returns
    -------
    tuple[pd.DataFrame, SersLoadReport]
        Concatenated wide dataframe and per-file load outcomes.
    """

    loaded_files: list[str] = []
    skipped_files: list[tuple[str, str]] = []
    dfs: List[pd.DataFrame] = []
    for file_path in files:
        try:
            df = _load_signal_file(file_path)
            if df.empty:
                continue
            if serotypes is not None:
                file_sero = str(df["serotype"].iloc[0]).strip().upper()
                allowed = {str(s).strip().upper() for s in serotypes}
                if file_sero not in allowed:
                    continue
            dfs.append(df)
            loaded_files.append(file_path.name)
        except Exception as exc:
            message = _user_facing_load_error(file_path.name, exc)
            logger.warning(
                "Skipping file %s: %s",
                file_path.name,
                message,
                exc_info=not isinstance(exc, ValueError),
            )
            skipped_files.append((file_path.name, message))

    wide_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    report = SersLoadReport(
        loaded_files=tuple(loaded_files),
        skipped_files=tuple(skipped_files),
    )
    return wide_df, report


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

    wide_df, _report = _load_sers_data_batch(files, serotypes=serotypes)
    return wide_df


def _get_raman_columns(df: pd.DataFrame) -> list:
    """Return sorted list of Raman shift column names (rs_* cols)."""
    rs_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(RS_COL_PREFIX)]
    return sorted(rs_cols, key=lambda c: float(c[len(RS_COL_PREFIX) :]))


def get_signals_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extract (n_samples, n_features) array from wide DataFrame for ML.

    Returns:
        Intensity matrix; rows = samples, columns = Raman shifts (sorted).
    """
    rs_cols = _get_raman_columns(df)
    if not rs_cols:
        raise ValueError("DataFrame has no Raman intensity columns; expected wide format")
    return df[rs_cols].values


def get_raman_shift(df: pd.DataFrame) -> np.ndarray:
    """Return the Raman shift (wavenumber) array for the spectral grid."""
    rs_cols = _get_raman_columns(df)
    if not rs_cols:
        raise ValueError("DataFrame has no Raman intensity columns; expected wide format")
    return np.array([float(c[len(RS_COL_PREFIX) :]) for c in rs_cols])


def get_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return all non-spectral columns of a wide frame with a positional index."""
    columns = [c for c in df.columns if not (isinstance(c, str) and c.startswith(RS_COL_PREFIX))]
    return df[columns].reset_index(drop=True)


def wide_to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide data to long format, retaining all non-spectral metadata.
    """
    rs_cols = _get_raman_columns(df)
    id_cols = [c for c in df.columns if c not in rs_cols]
    tidy = df.melt(
        id_vars=id_cols,
        value_vars=rs_cols,
        var_name="_rs_col",
        value_name="intensity",
    )
    tidy["raman_shift"] = tidy["_rs_col"].str[len(RS_COL_PREFIX) :].astype(float)
    return tidy.drop(columns=["_rs_col"])


def load_sers_data_as_wide_and_tidy(
    paths: Union[str, Path, List[Union[str, Path]]],
    *,
    serotypes: Optional[List[str]] = None,
    pattern: str = DEFAULT_PATTERN,
) -> tuple[pd.DataFrame, pd.DataFrame, SersLoadReport]:
    """
    Load SERS data and return both wide and tidy formats.

    Args:
        paths: File/folder path or list of paths.
        serotypes: If provided, only load matching serotypes.
        pattern: Glob pattern for folder scan (default: *.xlsx).

    Returns:
        Tuple of (wide_df, tidy_df, load_report). Dataframes are empty when
        every file fails to load.
    """
    files = _collect_files(paths, pattern)
    if not files:
        return pd.DataFrame(), pd.DataFrame(), SersLoadReport()

    wide, report = _load_sers_data_batch(files, serotypes=serotypes)
    if wide.empty:
        return wide, wide, report
    tidy = wide_to_tidy(wide)
    return wide, tidy, report


def count_unique_spectra(df: pd.DataFrame) -> int:
    """
    Count unique spectrum traces (filename + signal_index pairs) in a DataFrame.

    Args:
        df: Tidy or wide DataFrame with filename and signal_index columns.

    Returns:
        Number of unique spectra.
    """
    if df.empty or "filename" not in df.columns or "signal_index" not in df.columns:
        return 0
    return len(df.drop_duplicates(subset=["filename", "signal_index"]))
