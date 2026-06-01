"""Unit tests for TXT-to-Excel setup template and reload persistence helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APPS_DIR = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from txt_to_excel import (  # noqa: E402
    DATE_METADATA_FIELD_NUMBERS,
    METADATA_FIELD_SPECS,
    METADATA_WIDGET_KEYS,
    NUMERIC_METADATA_FIELD_NUMBERS,
    OPTIONAL_METADATA_FIELD_NUMBERS,
    RELOAD_CLEAR_WIDGET_KEYS,
    RELOAD_PERSIST_WIDGET_KEYS,
    TIME_METADATA_FIELD_NUMBERS,
    TEMPLATE_VERSION,
    _apply_template_import_to_values,
    _build_template_export_payload,
    _clear_reload_fields_in_snapshot,
    _coerce_metadata_widget_state_value,
    _format_metadata_date,
    _format_metadata_time,
    _is_template_field_exportable,
    _merge_widget_values_into_snapshot,
    _merged_preview_legend_labels,
    _normalize_legacy_preview_label,
    _parse_metadata_date,
    _parse_metadata_time,
    _serialize_template_field_value,
)


def test_reload_clear_keys_match_run_specific_metadata_fields() -> None:
    """Reload should clear run-specific metadata widgets only."""

    expected = {
        "txt2excel_meta_sensor_id",
        "txt2excel_meta_test_id",
        "txt2excel_meta_connection_id",
        "txt2excel_meta_serotype",
        "txt2excel_meta_testing_time",
        "txt2excel_meta_notes",
    }
    assert RELOAD_CLEAR_WIDGET_KEYS == expected


def test_reload_persist_keys_exclude_clear_set() -> None:
    """Instrument, date, operator, and rinsate widgets are not cleared on reload."""

    assert "txt2excel_meta_disk_diameter_nm" in RELOAD_PERSIST_WIDGET_KEYS
    assert "txt2excel_meta_sensor_model" in RELOAD_PERSIST_WIDGET_KEYS
    assert "txt2excel_meta_date" in RELOAD_PERSIST_WIDGET_KEYS
    assert "txt2excel_meta_operator" in RELOAD_PERSIST_WIDGET_KEYS
    assert "txt2excel_meta_rinsate_type" in RELOAD_PERSIST_WIDGET_KEYS
    assert "txt2excel_meta_sensor_id" not in RELOAD_PERSIST_WIDGET_KEYS
    assert RELOAD_PERSIST_WIDGET_KEYS == METADATA_WIDGET_KEYS - RELOAD_CLEAR_WIDGET_KEYS


def test_persistent_snapshot_survives_reload_clear_simulation() -> None:
    """Snapshot keeps instrument metadata while run-specific fields are blanked."""

    widget_values = {
        "txt2excel_meta_disk_diameter_nm": "200",
        "txt2excel_meta_sensor_model": "Model-A",
        "txt2excel_meta_sensor_id": "S-1",
        "txt2excel_meta_date": "2026-05-29",
        "txt2excel_meta_operator": "Lab Tech",
    }
    snapshot = _merge_widget_values_into_snapshot({}, widget_values)
    snapshot = _clear_reload_fields_in_snapshot(snapshot)
    assert snapshot["txt2excel_meta_disk_diameter_nm"] == "200"
    assert snapshot["txt2excel_meta_sensor_model"] == "Model-A"
    assert snapshot["txt2excel_meta_date"] == "2026-05-29"
    assert snapshot["txt2excel_meta_operator"] == "Lab Tech"
    assert snapshot["txt2excel_meta_sensor_id"] == ""


def test_optional_notes_field_number() -> None:
    """Notes is field 16 and optional."""

    notes_specs = [spec for spec in METADATA_FIELD_SPECS if spec[1] == "Notes"]
    assert notes_specs == [(16, "Notes", "txt2excel_meta_notes")]
    assert OPTIONAL_METADATA_FIELD_NUMBERS == frozenset({16})


def test_numeric_metadata_fields_are_geometry_and_acquisition() -> None:
    """Fields 1–4 and 6–7 require numeric workbook values."""

    assert NUMERIC_METADATA_FIELD_NUMBERS == frozenset({1, 2, 3, 4, 6, 7})


def test_date_and_time_metadata_field_numbers() -> None:
    """Fields 13 and 14 are validated as date and time."""

    assert DATE_METADATA_FIELD_NUMBERS == frozenset({13})
    assert TIME_METADATA_FIELD_NUMBERS == frozenset({14})


def test_metadata_field_specs_follow_grouped_order() -> None:
    """Excel/UI order is 5+2+3+2+3 fields across three five-column rows, then notes."""

    from txt_to_excel import (
        METADATA_LOGICAL_GROUPS,
        METADATA_UI_ROWS,
        _SEPARATOR_AFTER_METADATA_NUMBERS,
    )

    labels = [label for _, label, _ in METADATA_FIELD_SPECS]
    assert labels == [
        "Disk Diameter (nm)",
        "Periodicity (µm)",
        "Thickness (nm)",
        "Core Diameter (µm)",
        "Sensor Model",
        "Integration Time (ms)",
        "Scan Average",
        "Sensor ID",
        "Test ID",
        "Connection ID",
        "Serotype",
        "Rinsate Type",
        "Date",
        "Testing Time",
        "Operator",
        "Notes",
    ]
    flattened_groups = [
        field_number for group_fields in METADATA_LOGICAL_GROUPS for field_number in group_fields
    ]
    assert flattened_groups == list(range(1, 17))
    assert METADATA_UI_ROWS == ((1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (11, 12, 13, 14, 15))
    assert _SEPARATOR_AFTER_METADATA_NUMBERS == frozenset({5, 7, 10, 12, 15, 16})


def test_parse_and_format_metadata_date_and_time() -> None:
    """Date and time helpers normalize valid values and reject invalid input."""

    from datetime import date, time

    assert _parse_metadata_date("2026-05-29") == date(2026, 5, 29)
    assert _parse_metadata_date("05/29/2026") == date(2026, 5, 29)
    assert _parse_metadata_date("not-a-date") is None
    assert _parse_metadata_time("2:30 PM") == time(14, 30)
    assert _parse_metadata_time("2:30pm") == time(14, 30)
    assert _parse_metadata_time("2 PM") == time(14, 0)
    assert _parse_metadata_time("2pm") == time(14, 0)
    assert _parse_metadata_time("14:30") == time(14, 30)
    assert _parse_metadata_time("morning") is None
    assert _format_metadata_date(date(2026, 5, 29)) == "2026-05-29"
    assert _format_metadata_time(time(14, 30)) == "02:30 PM"
    assert _format_metadata_time(time(14, 30, 15)) == "02:30:15 PM"
    assert _format_metadata_time(time(0, 15)) == "12:15 AM"
    assert _format_metadata_time("1:30 PM") == "01:30 PM"


def test_merged_preview_legend_labels_use_colon_format() -> None:
    """Preview legend labels separate targets and actuals with colons."""

    assert _normalize_legacy_preview_label("Target 1000 / Actual 995") == (
        "Target: 1000 / Actual: 995"
    )
    payload = {
        "target_concentrations": [1000, 0],
        "actual_concentrations": [995, 0],
    }
    assert _merged_preview_legend_labels(payload) == [
        "Target: 1000 / Actual: 995",
        "Target: 0 / Actual: 0",
    ]


def test_merged_preview_legend_labels_blank_target() -> None:
    """Preview legend uses an em dash when target CFU is omitted."""

    payload = {
        "target_concentrations": [""],
        "actual_concentrations": [1000],
        "source_txt_filenames": ["paa_sample.txt"],
    }
    assert _merged_preview_legend_labels(payload) == [
        "Target: — / Actual: 1000 (paa_sample.txt)",
    ]


def test_coerce_testing_time_widget_accepts_parsed_template_strings() -> None:
    """Testing-time widget state stores a ``time`` object for ``time_input``."""

    from datetime import time

    assert _coerce_metadata_widget_state_value(14, "10:45 AM") == time(10, 45)
    assert _coerce_metadata_widget_state_value(14, time(14, 30)) == time(14, 30)
    assert _coerce_metadata_widget_state_value(14, None) is None
    assert _coerce_metadata_widget_state_value(14, "morning") is None


def test_build_template_export_payload_includes_only_selected_valid_fields() -> None:
    """Export includes checked fields with valid values and skips empty numerics."""

    field_values = {
        "txt2excel_meta_disk_diameter_nm": "200",
        "txt2excel_meta_periodicity_um": "",
        "txt2excel_meta_sensor_model": "Model-X",
        "txt2excel_meta_notes": "",
    }
    selected = frozenset(
        {
            "txt2excel_meta_disk_diameter_nm",
            "txt2excel_meta_periodicity_um",
            "txt2excel_meta_sensor_model",
            "txt2excel_meta_notes",
        }
    )
    payload = _build_template_export_payload(field_values, selected)
    assert payload is not None
    assert payload["version"] == TEMPLATE_VERSION
    assert payload["fields"] == {
        "txt2excel_meta_disk_diameter_nm": 200,
        "txt2excel_meta_sensor_model": "Model-X",
    }


def test_build_template_export_payload_returns_none_when_nothing_valid() -> None:
    """Export returns None when no selected field has a valid value."""

    field_values = {"txt2excel_meta_disk_diameter_nm": "not-a-number"}
    selected = frozenset({"txt2excel_meta_disk_diameter_nm"})
    assert _build_template_export_payload(field_values, selected) is None


def test_template_export_rejects_invalid_date_and_time_values() -> None:
    """Template export skips invalid date and time values."""

    field_values = {
        "txt2excel_meta_date": "not-a-date",
        "txt2excel_meta_testing_time": "morning",
    }
    selected = frozenset(field_values)
    assert _build_template_export_payload(field_values, selected) is None
    assert _is_template_field_exportable(13, "2026-05-29")
    assert not _is_template_field_exportable(13, "bad-date")
    assert _is_template_field_exportable(14, "10:45 AM")
    assert not _is_template_field_exportable(14, "morning")
    assert _serialize_template_field_value(13, "2026-05-29") == "2026-05-29"
    assert _serialize_template_field_value(14, "10:45 AM") == "10:45 AM"
    assert _serialize_template_field_value(14, "10:45") == "10:45 AM"


def test_apply_template_import_merges_valid_fields() -> None:
    """Import overwrites only keys present in the template payload."""

    payload = {
        "version": TEMPLATE_VERSION,
        "fields": {
            "txt2excel_meta_disk_diameter_nm": 250,
            "txt2excel_meta_sensor_model": "Imported",
            "txt2excel_meta_unknown": "skip-me",
        },
    }
    before = {key: "" for key in METADATA_WIDGET_KEYS}
    before["txt2excel_meta_operator"] = "Existing Operator"
    updated, warnings = _apply_template_import_to_values(payload, before)
    assert updated["txt2excel_meta_disk_diameter_nm"] == "250"
    assert updated["txt2excel_meta_sensor_model"] == "Imported"
    assert updated["txt2excel_meta_operator"] == "Existing Operator"
    assert any("Unknown template field" in message for message in warnings)


def test_apply_template_import_rejects_bad_version() -> None:
    """Import warns and leaves values unchanged for unsupported template versions."""

    before = {"txt2excel_meta_sensor_model": "Keep Me"}
    updated, warnings = _apply_template_import_to_values({"version": 99, "fields": {}}, before)
    assert updated == before
    assert any("Unsupported template version" in message for message in warnings)


def test_apply_template_import_validates_date_and_time() -> None:
    """Import normalizes valid date/time values and warns on invalid ones."""

    before = {key: "" for key in METADATA_WIDGET_KEYS}
    updated, warnings = _apply_template_import_to_values(
        {
            "version": TEMPLATE_VERSION,
            "fields": {
                "txt2excel_meta_date": "2026-06-01",
                "txt2excel_meta_testing_time": "10:45 AM",
            },
        },
        before,
    )
    assert updated["txt2excel_meta_date"] == "2026-06-01"
    assert updated["txt2excel_meta_testing_time"] == "10:45 AM"

    before = {key: "" for key in METADATA_WIDGET_KEYS}
    _, warnings = _apply_template_import_to_values(
        {
            "version": TEMPLATE_VERSION,
            "fields": {
                "txt2excel_meta_date": "June 1",
                "txt2excel_meta_testing_time": "ten",
            },
        },
        before,
    )
    assert any("Invalid date value" in message for message in warnings)
    assert any("Invalid time value" in message for message in warnings)


@pytest.mark.parametrize(
    ("field_number", "raw_value", "expected"),
    [
        (1, "100", True),
        (1, "", False),
        (1, "abc", False),
        (5, "Model-A", True),
        (5, "", False),
        (7, "4", True),
        (7, "abc", False),
        (16, "", False),
        (16, "Optional note", True),
    ],
)
def test_is_template_field_exportable(field_number: int, raw_value: str, expected: bool) -> None:
    """Template exportability follows numeric, text, and optional Notes rules."""

    assert _is_template_field_exportable(field_number, raw_value) is expected
