"""Tests for per-signal File Name and Special Treatment rows in embedded Excel I/O."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

APPS_DIR = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from txt_to_excel import (  # noqa: E402
    DATE_METADATA_FIELD_NUMBERS,
    METADATA_FIELD_SPECS,
    TIME_METADATA_FIELD_NUMBERS,
    _build_embedded_workbook_rows,
    _embedded_workbook_to_excel_bytes,
    _format_merged_preview_legend_label,
    _parse_optional_number,
)

from sensd_sers_analysis.data.io import _load_signal_file  # noqa: E402


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


def _write_minimal_embedded_xlsx(tmp_path: Path, **row_kwargs) -> Path:
    """Build and write a minimal embedded workbook for loader tests."""

    raman_shift = np.array([100.0, 101.0])
    intensity_columns = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    defaults = {
        "target_concentrations": ["", 1000.0],
        "actual_concentrations": [500.0, 1000.0],
        "source_txt_filenames": ["heat_kill.txt", "live_1000.txt"],
        "special_treatments": ["heat kill", ""],
    }
    defaults.update(row_kwargs)
    rows = _build_embedded_workbook_rows(
        _minimal_workbook_metadata(),
        raman_shift=raman_shift,
        intensity_columns=intensity_columns,
        **defaults,
    )
    path = tmp_path / "embedded_test.xlsx"
    path.write_bytes(_embedded_workbook_to_excel_bytes(rows))
    return path


def test_parse_optional_number_allows_blank() -> None:
    """Optional numeric fields accept empty strings but reject non-numeric text."""

    assert _parse_optional_number("") is None
    assert _parse_optional_number("   ") is None
    assert _parse_optional_number("1000") == 1000.0
    assert _parse_optional_number("PAA") is None


def test_format_merged_preview_legend_shows_blank_target() -> None:
    """Preview legend marks missing target CFU with an em dash."""

    label = _format_merged_preview_legend_label(
        "",
        500,
        source_txt_filename="heat_kill.txt",
    )
    assert label == "Target: — / Actual: 500 (heat_kill.txt)"


def test_load_signal_file_reads_per_signal_rows(tmp_path: Path) -> None:
    """Loader exposes source TXT filename and special treatment per signal."""

    path = _write_minimal_embedded_xlsx(tmp_path)
    wide = _load_signal_file(path)
    assert list(wide["source_txt_filename"]) == ["heat_kill.txt", "live_1000.txt"]
    assert list(wide["special_treatment"]) == ["heat kill", ""]
    assert list(wide["concentration"]) == [500.0, 1000.0]


def test_load_signal_file_without_per_signal_rows(tmp_path: Path) -> None:
    """Legacy workbooks without new rows still load with empty provenance fields."""

    raman_shift = np.array([200.0])
    intensity_columns = [np.array([5.0])]
    rows = _build_embedded_workbook_rows(
        _minimal_workbook_metadata(),
        target_concentrations=[100.0],
        actual_concentrations=[100.0],
        source_txt_filenames=[""],
        special_treatments=[""],
        raman_shift=raman_shift,
        intensity_columns=intensity_columns,
    )
    # Legacy layout: omit File Name and Special Treatment rows (target/actual only).
    n_meta = len(METADATA_FIELD_SPECS)
    legacy_rows = rows[:n_meta] + rows[n_meta + 2 : n_meta + 4] + rows[n_meta + 4 :]
    path = tmp_path / "legacy.xlsx"
    path.write_bytes(_embedded_workbook_to_excel_bytes(legacy_rows))

    wide = _load_signal_file(path)
    assert wide["source_txt_filename"].iloc[0] == ""
    assert wide["special_treatment"].iloc[0] == ""
