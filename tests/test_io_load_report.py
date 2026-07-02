"""Tests for resilient embedded Excel loading and user-facing load reports."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

APPS_DIR = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from txt_to_excel import (  # noqa: E402
    DATE_METADATA_FIELD_NUMBERS,
    METADATA_FIELD_SPECS,
    TIME_METADATA_FIELD_NUMBERS,
    _build_embedded_workbook_rows,
    _embedded_workbook_to_excel_bytes,
)

from sensd_sers_analysis.data.io import (  # noqa: E402
    SersLoadReport,
    _load_signal_file,
    load_sers_data_as_wide_and_tidy,
)


def _minimal_workbook_metadata() -> dict[str, object]:
    """Return minimal valid metadata for embedded workbook row building."""

    values: dict[str, object] = {}
    for field_number, excel_label, _ in METADATA_FIELD_SPECS:
        if field_number <= 6:
            values[excel_label] = float(field_number)
        elif field_number in DATE_METADATA_FIELD_NUMBERS:
            values[excel_label] = "2026-06-01"
        elif field_number in TIME_METADATA_FIELD_NUMBERS:
            values[excel_label] = "10:00 AM"
        else:
            values[excel_label] = f"meta-{field_number}"
    return values


def _write_minimal_embedded_xlsx(tmp_path: Path, *, extra_top_rows: int = 0) -> Path:
    """Build and write a minimal embedded workbook for loader tests."""

    raman_shift = np.array([100.0, 101.0])
    intensity_columns = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    rows = _build_embedded_workbook_rows(
        _minimal_workbook_metadata(),
        target_concentrations=["", 1000.0],
        actual_concentrations=[500.0, 1000.0],
        source_txt_filenames=["heat_kill.txt", "live_1000.txt"],
        special_treatments=["heat kill", ""],
        raman_shift=raman_shift,
        intensity_columns=intensity_columns,
    )
    for _ in range(extra_top_rows):
        rows.insert(0, [np.nan, np.nan] + [""] * (len(rows[0]) - 2))

    path = tmp_path / "embedded_test.xlsx"
    path.write_bytes(_embedded_workbook_to_excel_bytes(rows))
    return path


def _write_invalid_workbook(tmp_path: Path) -> Path:
    """Write a workbook missing the concentration row."""

    buffer = BytesIO()
    pd.DataFrame([["Sensor ID", "S1"], ["Serotype", "ST"]]).to_excel(
        buffer,
        index=False,
        header=False,
    )
    path = tmp_path / "invalid.xlsx"
    path.write_bytes(buffer.getvalue())
    return path


def test_load_signal_file_tolerates_blank_metadata_rows(tmp_path: Path) -> None:
    """Blank column-A rows above metadata should not crash the parser."""

    path = _write_minimal_embedded_xlsx(tmp_path, extra_top_rows=2)
    wide = _load_signal_file(path)
    assert not wide.empty
    assert list(wide["concentration"]) == [500.0, 1000.0]


def test_load_sers_data_as_wide_and_tidy_reports_invalid_file(tmp_path: Path) -> None:
    """Invalid workbooks are skipped with a filename in the load report."""

    invalid_path = _write_invalid_workbook(tmp_path)
    wide_df, tidy_df, report = load_sers_data_as_wide_and_tidy([str(invalid_path)])

    assert wide_df.empty
    assert tidy_df.empty
    assert report.n_loaded == 0
    assert report.n_skipped == 1
    filename, message = report.skipped_files[0]
    assert filename == "invalid.xlsx"
    assert "invalid.xlsx" in message
    assert "Concentration row not found" in message


def test_load_sers_data_as_wide_and_tidy_partial_batch(tmp_path: Path) -> None:
    """Good files still load when another file in the batch is invalid."""

    good_path = _write_minimal_embedded_xlsx(tmp_path)
    invalid_path = _write_invalid_workbook(tmp_path)
    wide_df, tidy_df, report = load_sers_data_as_wide_and_tidy([str(good_path), str(invalid_path)])

    assert not wide_df.empty
    assert not tidy_df.empty
    assert report.n_loaded == 1
    assert report.n_skipped == 1
    assert report.loaded_files == ("embedded_test.xlsx",)
    assert report.skipped_files[0][0] == "invalid.xlsx"


def test_sers_load_report_defaults() -> None:
    """Empty load reports expose zero counts."""

    report = SersLoadReport()
    assert report.n_loaded == 0
    assert report.n_skipped == 0
