"""
Instrument TXT to Excel merger — Streamlit prep utility.

Converts raw Raman instrument ``.txt`` exports into embedded-metadata Excel
workbooks for the SERS analysis tool. Lives entirely under ``apps/``.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import date, datetime, time
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

logger = logging.getLogger(__name__)

APP_MODE_KEY = "_app_mode"
APP_MODE_ANALYSIS = "analysis"
APP_MODE_PREP = "prep"

MERGED_PREVIEW_KEY = "_txt2excel_merged_preview"
EXPORT_BYTES_KEY = "_txt2excel_export_bytes"
EXPORT_FILENAME_KEY = "_txt2excel_export_filename"
PREP_UPLOADER_RESET_KEY = "_txt2excel_uploader_reset"
TEMPLATE_IMPORT_UPLOADER_RESET_KEY = "_txt2excel_template_import_reset"
TEMPLATE_IMPORT_FEEDBACK_KEY = "_txt2excel_template_import_feedback"
TEMPLATE_IMPORT_PENDING_VALUES_KEY = "_txt2excel_template_import_pending_values"
TEMPLATE_EXPORT_SELECT_ALL_KEY = "txt2excel_template_export_select_all"
TEMPLATE_EXPORT_FILENAME = "SERS_metadata_preset.json"
TESTING_TIME_WIDGET_KEY = "txt2excel_meta_testing_time"
TEMPLATE_VERSION = 1
DEFAULT_MIN_SHIFT = 560.9

TARGET_CONCENTRATION_LABEL = "Target Concentration (CFU/mL)"
ACTUAL_CONCENTRATION_LABEL = "Actual Concentration (CFU/mL)"
PREP_TARGET_CONCENTRATION_HEADER = "Target Concentration"
PREP_ACTUAL_CONCENTRATION_HEADER = "Actual Concentration"
INTENSITY_HEADER = "Relative Light intensity (a.u)"
RAMAN_SHIFT_HEADER = "Raman Shift"

# Metadata field numbers whose column-B values must be numeric in the workbook.
NUMERIC_METADATA_FIELD_NUMBERS = frozenset({1, 2, 3, 4, 5, 6})
DATE_METADATA_FIELD_NUMBERS = frozenset({12})
TIME_METADATA_FIELD_NUMBERS = frozenset({13})
OPTIONAL_METADATA_FIELD_NUMBERS = frozenset({16})
METADATA_FIELD_SPECS: tuple[tuple[int, str, str], ...] = (
    (1, "Disk Diameter (nm)", "txt2excel_meta_disk_diameter_nm"),
    (2, "Periodicity (µm)", "txt2excel_meta_periodicity_um"),
    (3, "Thickness (nm)", "txt2excel_meta_thickness_nm"),
    (4, "Core Diameter (µm)", "txt2excel_meta_core_diameter_um"),
    (5, "Integration Time (ms):", "txt2excel_meta_integration_time_ms"),
    (6, "Scan Average:", "txt2excel_meta_scan_average"),
    (7, "Sensor Model", "txt2excel_meta_sensor_model"),
    (8, "Sensor ID", "txt2excel_meta_sensor_id"),
    (9, "Test ID", "txt2excel_meta_test_id"),
    (10, "Connection ID", "txt2excel_meta_connection_id"),
    (11, "Serotype", "txt2excel_meta_serotype"),
    (12, "Date", "txt2excel_meta_date"),
    (13, "Testing Time", "txt2excel_meta_testing_time"),
    (14, "Operator", "txt2excel_meta_operator"),
    (15, "Rinsate Type", "txt2excel_meta_rinsate_type"),
    (16, "Notes", "txt2excel_meta_notes"),
)
METADATA_WIDGET_KEYS = frozenset(widget_key for _, _, widget_key in METADATA_FIELD_SPECS)
RELOAD_CLEAR_WIDGET_KEYS = frozenset(
    {
        "txt2excel_meta_sensor_id",
        "txt2excel_meta_test_id",
        "txt2excel_meta_connection_id",
        "txt2excel_meta_serotype",
        "txt2excel_meta_testing_time",
        "txt2excel_meta_notes",
    }
)
RELOAD_PERSIST_WIDGET_KEYS = frozenset(METADATA_WIDGET_KEYS - RELOAD_CLEAR_WIDGET_KEYS)
PERSISTENT_METADATA_SNAPSHOT_KEY = "_txt2excel_persistent_metadata"
RESTORE_METADATA_AFTER_RELOAD_KEY = "_txt2excel_restore_metadata_after_reload"


def enter_prep_mode() -> None:
    """Switch the app shell to the TXT-to-Excel prep utility."""

    st.session_state[APP_MODE_KEY] = APP_MODE_PREP


def enter_analysis_mode() -> None:
    """Return to the main SERS analysis explorer."""

    st.session_state[APP_MODE_KEY] = APP_MODE_ANALYSIS
    st.session_state.pop(MERGED_PREVIEW_KEY, None)
    st.session_state.pop(EXPORT_BYTES_KEY, None)
    st.session_state.pop(EXPORT_FILENAME_KEY, None)


def _clear_session_keys_by_prefix(prefix: str) -> None:
    """Remove all session-state keys that start with ``prefix``."""

    for key in list(st.session_state.keys()):
        if key.startswith(prefix):
            del st.session_state[key]


def _merge_widget_values_into_snapshot(
    snapshot: dict[str, str],
    widget_values: dict[str, str],
    *,
    persist_keys: frozenset[str] = RELOAD_PERSIST_WIDGET_KEYS,
) -> dict[str, str]:
    """
    Merge widget values into a persistent metadata snapshot.

    Streamlit drops widget session keys when inputs are not rendered (e.g. after
    Reload Data clears uploads). The snapshot survives across those runs.
    """

    updated = dict(snapshot)
    for key in persist_keys:
        if key in widget_values:
            updated[key] = widget_values[key]
    return updated


def _clear_reload_fields_in_snapshot(
    snapshot: dict[str, str],
    *,
    clear_keys: frozenset[str] = RELOAD_CLEAR_WIDGET_KEYS,
) -> dict[str, str]:
    """Blank run-specific metadata keys in the persistent snapshot."""

    updated = dict(snapshot)
    for key in clear_keys:
        updated[key] = ""
    return updated


def _sync_persistent_metadata_snapshot(widget_values: dict[str, str] | None = None) -> None:
    """Save persistent metadata widget values into the cross-run snapshot."""

    values = (
        widget_values if widget_values is not None else _collect_metadata_values_by_widget_key()
    )
    snapshot = st.session_state.get(PERSISTENT_METADATA_SNAPSHOT_KEY, {})
    st.session_state[PERSISTENT_METADATA_SNAPSHOT_KEY] = _merge_widget_values_into_snapshot(
        snapshot,
        values,
    )


def _restore_persistent_metadata_widgets() -> None:
    """Restore persistent metadata widgets from the snapshot before rendering inputs."""

    snapshot = st.session_state.get(PERSISTENT_METADATA_SNAPSHOT_KEY, {})
    for field_number, _, key in METADATA_FIELD_SPECS:
        if key in RELOAD_PERSIST_WIDGET_KEYS and key in snapshot:
            st.session_state[key] = _coerce_metadata_widget_state_value(field_number, snapshot[key])


def _apply_pending_template_import_values() -> None:
    """Apply deferred template-import values before metadata widgets are rendered."""

    pending_values = st.session_state.pop(TEMPLATE_IMPORT_PENDING_VALUES_KEY, None)
    if not isinstance(pending_values, dict):
        return
    for widget_key, value in pending_values.items():
        if widget_key not in METADATA_WIDGET_KEYS:
            continue
        field_number = _field_number_for_widget_key(widget_key)
        if field_number is None:
            continue
        st.session_state[widget_key] = _coerce_metadata_widget_state_value(field_number, value)
    _sync_persistent_metadata_snapshot(
        {
            widget_key: _metadata_widget_state_to_string(
                field_number, st.session_state.get(widget_key)
            )
            for widget_key in pending_values
            if widget_key in METADATA_WIDGET_KEYS
            for field_number in [_field_number_for_widget_key(widget_key)]
            if field_number is not None
        }
    )


def clear_prep_uploads() -> None:
    """
    Reset uploaded TXT files and derived export artifacts.

    Instrument metadata (fields 1–7), date, operator, and rinsate type persist;
    run-specific fields, concentrations, and Raman bounds reset for the next batch.
    """

    logger.info("Clearing prep upload state (Reload Data clicked)")
    widget_values = {
        widget_key: _metadata_widget_state_to_string(
            field_number,
            st.session_state.get(widget_key),
        )
        for field_number, _, widget_key in METADATA_FIELD_SPECS
        if widget_key in st.session_state
    }
    snapshot = st.session_state.get(PERSISTENT_METADATA_SNAPSHOT_KEY, {})
    snapshot = _merge_widget_values_into_snapshot(snapshot, widget_values)
    snapshot = _clear_reload_fields_in_snapshot(snapshot)
    st.session_state[PERSISTENT_METADATA_SNAPSHOT_KEY] = snapshot

    st.session_state.pop(MERGED_PREVIEW_KEY, None)
    st.session_state.pop(EXPORT_BYTES_KEY, None)
    st.session_state.pop(EXPORT_FILENAME_KEY, None)
    st.session_state.pop("txt2excel_file_signature", None)
    st.session_state.pop("txt2excel_min_shift", None)
    st.session_state.pop("txt2excel_max_shift", None)
    for widget_key in RELOAD_CLEAR_WIDGET_KEYS:
        field_number = _field_number_for_widget_key(widget_key)
        if field_number in DATE_METADATA_FIELD_NUMBERS:
            st.session_state[widget_key] = None
        else:
            st.session_state[widget_key] = ""
    _clear_session_keys_by_prefix("txt2excel_target_")
    _clear_session_keys_by_prefix("txt2excel_actual_")
    st.session_state[RESTORE_METADATA_AFTER_RELOAD_KEY] = True
    st.session_state[PREP_UPLOADER_RESET_KEY] = str(uuid.uuid4())


def render_prep_entry_in_sidebar(sidebar: DeltaGenerator) -> None:
    """
    Render a compact link-style control above the analysis data-loading section.

    Parameters
    ----------
    sidebar:
        Streamlit sidebar container.
    """

    sidebar.button(
        "Convert TXT → Excel",
        key="enter_txt_to_excel_mode",
        on_click=enter_prep_mode,
        use_container_width=True,
    )
    sidebar.markdown("---")


def _extract_cfu_sort_key(filename: str) -> int:
    """Extract the first integer in a filename for natural dilution ordering."""

    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else int(1e9)


def _parse_txt_content(content: str) -> pd.DataFrame:
    """
    Parse one instrument TXT export into Raman shift and intensity columns.

    Parameters
    ----------
    content:
        UTF-8 text of the instrument export.

    Returns
    -------
    pd.DataFrame
        Columns ``RamanShift`` and ``Value``.
    """

    df = pd.read_csv(
        StringIO(content),
        sep="\t",
        comment=">",
        header=None,
        names=["RamanShift", "Value"],
    )
    df["RamanShift"] = pd.to_numeric(df["RamanShift"], errors="coerce")
    return df.dropna(subset=["RamanShift", "Value"]).reset_index(drop=True)


def _validate_common_shift(
    file_contents: dict[str, str],
    *,
    atol: float = 1e-4,
) -> tuple[np.ndarray | None, str | None]:
    """
    Verify all uploaded TXT files share the same Raman shift grid.

    Returns
    -------
    tuple
        ``(common_shift, error_message)`` — shift array when valid, else ``None`` and message.
    """

    common_shift: np.ndarray | None = None
    for name, content in file_contents.items():
        shifts = _parse_txt_content(content)["RamanShift"].to_numpy()
        if common_shift is None:
            common_shift = shifts
            continue
        if not np.allclose(common_shift, shifts, atol=atol):
            return None, f"Raman shift mismatch in file: {name}"
    return common_shift, None


def _merge_txt_spectra(
    file_contents: dict[str, str],
    *,
    min_shift: float | None,
    max_shift: float | None,
) -> tuple[np.ndarray | None, list[np.ndarray] | None, str | None]:
    """
    Merge parsed TXT spectra into aligned Raman shift and intensity columns.

    Parameters
    ----------
    file_contents:
        Mapping of filename to UTF-8 file text, in column order.
    min_shift:
        Optional lower Raman shift bound (cm⁻¹).
    max_shift:
        Optional upper Raman shift bound (cm⁻¹).

    Returns
    -------
    tuple
        ``(raman_shift, intensity_columns, error_message)``.
    """

    names = list(file_contents.keys())
    if not names:
        return None, None, "No TXT files provided."

    common_shift, shift_error = _validate_common_shift(file_contents)
    if shift_error:
        return None, None, shift_error
    if common_shift is None:
        return None, None, "No Raman shift data found."

    min_final = float(min_shift) if min_shift is not None else float(common_shift.min())
    max_final = float(max_shift) if max_shift is not None else float(common_shift.max())
    if min_final >= max_final:
        return None, None, "Min Raman Shift must be less than Max."

    raman_shift = common_shift[(common_shift >= min_final) & (common_shift <= max_final)]
    intensity_columns: list[np.ndarray] = []

    for name in names:
        df = _parse_txt_content(file_contents[name])
        trimmed = df[(df["RamanShift"] >= min_final) & (df["RamanShift"] <= max_final)].reset_index(
            drop=True
        )
        if len(trimmed) != len(raman_shift):
            return None, None, f"Truncated row count mismatch for file: {name}"
        intensity_columns.append(trimmed["Value"].to_numpy(dtype=float))

    return raman_shift, intensity_columns, None


def _parse_required_number(raw: str) -> float | None:
    """Parse a required numeric field; returns ``None`` when empty or non-numeric."""

    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _to_excel_number(value: float) -> int | float:
    """Coerce a numeric value for Excel cells (integers stored without decimals)."""

    if value.is_integer():
        return int(value)
    return value


def _parse_metadata_date(raw: str) -> date | None:
    """Parse a metadata date string in ISO or common US format."""

    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return date.fromisoformat(stripped)
    except ValueError:
        pass
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(stripped, fmt).date()
        except ValueError:
            continue
    return None


def _normalize_ampm_period(raw_period: str) -> str:
    """Return ``AM`` or ``PM`` from a case-insensitive period token."""

    token = raw_period.upper().replace(".", "")
    return "AM" if token.startswith("A") else "PM"


def _ampm_parse_candidates(raw: str) -> list[str]:
    """
    Build normalized 12-hour time strings with an explicit AM/PM suffix.

    Accepts compact input such as ``230pm`` or ``2:30 PM`` and expands it into
    ``strptime``-compatible candidates.
    """

    text = re.sub(r"\s+", " ", raw.strip())
    if not text:
        return []

    ampm_match = re.search(r"(a\.?m\.?|p\.?m\.?)\.?\s*$", text, re.IGNORECASE)
    if not ampm_match:
        return [text]

    period = _normalize_ampm_period(ampm_match.group(1))
    time_part = text[: ampm_match.start()].strip()
    if not time_part:
        return [f"12 {period}"]

    candidates: list[str] = []
    if ":" in time_part:
        candidates.append(f"{time_part} {period}")
    else:
        digits = re.sub(r"\D", "", time_part)
        if not digits:
            return []
        if len(digits) <= 2:
            candidates.append(f"{int(digits)} {period}")
        elif len(digits) == 3:
            candidates.append(f"{digits[0]}:{digits[1:]} {period}")
        else:
            hour_digits = digits[:2]
            minute_digits = digits[2:4]
            if int(hour_digits) > 12:
                candidates.append(f"{digits[0]}:{digits[1:3]} {period}")
            else:
                candidates.append(f"{int(hour_digits)}:{minute_digits} {period}")
                if len(digits) >= 4:
                    candidates.append(f"{int(digits[0])}:{digits[1:3]} {period}")

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _parse_metadata_time(raw: str) -> time | None:
    """Parse a metadata time string in 12-hour AM/PM or legacy 24-hour format."""

    stripped = raw.strip()
    if not stripped:
        return None

    ampm_formats = ("%I:%M:%S %p", "%I:%M %p", "%I %p")
    for candidate in _ampm_parse_candidates(stripped):
        for fmt in ampm_formats:
            if fmt == "%I %p" and ":" in candidate:
                continue
            if fmt.startswith("%I:%M") and candidate.count(":") != fmt.count(":"):
                continue
            try:
                return datetime.strptime(candidate, fmt).time()
            except ValueError:
                continue

    legacy_candidates = [stripped]
    masked = _format_time_digit_mask(stripped)
    if masked and masked not in legacy_candidates:
        legacy_candidates.append(masked)
    for candidate in legacy_candidates:
        for fmt in ("%H:%M:%S", "%H:%M"):
            if fmt == "%H:%M" and len(candidate) != 5:
                continue
            try:
                return datetime.strptime(candidate, fmt).time()
            except ValueError:
                continue
    return None


def _format_time_digit_mask(raw: str) -> str:
    """
    Format up to four typed digits as progressive ``H:MM`` mask text.

    Non-digits are stripped. A colon is inserted automatically after the hour digits.
    When the first two digits exceed 12, a single-digit hour is used for 12-hour entry.
    """

    digits = re.sub(r"\D", "", raw)[:4]
    if not digits:
        return ""
    if len(digits) <= 2:
        return digits
    if len(digits) == 3:
        return f"{digits[0]}:{digits[1:]}"
    if int(digits[:2]) > 12:
        return f"{digits[0]}:{digits[1:3]}"
    return f"{digits[:2]}:{digits[2:]}"


def _zero_pad_ampm_clock(clock: str, *, period_suffix: bool = False) -> str:
    """Zero-pad hour and minute segments when the clock portion is complete."""

    stripped = clock.strip()
    if not stripped:
        return ""

    if ":" in stripped:
        hour_str, remainder = stripped.split(":", 1)
        if not hour_str.isdigit():
            return stripped
        if ":" in remainder:
            minute_str, second_str = remainder.split(":", 1)
            if (
                minute_str.isdigit()
                and second_str.isdigit()
                and len(minute_str) == 2
                and len(second_str) == 2
            ):
                return f"{int(hour_str):02d}:{int(minute_str):02d}:{int(second_str):02d}"
            return stripped
        minute_str = remainder
        if minute_str.isdigit() and len(minute_str) == 2:
            return f"{int(hour_str):02d}:{int(minute_str):02d}"
        if minute_str.isdigit() and len(minute_str) == 1:
            return f"{int(hour_str):02d}:{minute_str}"
        return stripped

    if stripped.isdigit() and period_suffix:
        return f"{int(stripped):02d}"
    return stripped


def _format_ampm_time_mask(raw: str) -> str:
    """Format progressive 12-hour clock text while preserving a trailing AM/PM suffix."""

    stripped = raw.strip()
    ampm_match = re.search(r"(a\.?m\.?|p\.?m\.?)\.?\s*$", stripped, re.IGNORECASE)
    suffix = ""
    time_part = stripped
    if ampm_match:
        suffix = f" {_normalize_ampm_period(ampm_match.group(1))}"
        time_part = stripped[: ampm_match.start()]

    masked = _format_time_digit_mask(time_part)
    if masked:
        masked = _zero_pad_ampm_clock(masked, period_suffix=bool(suffix))
    if not masked and not suffix:
        return ""
    return f"{masked}{suffix}".strip()


def _format_metadata_time_display(parsed: time) -> str:
    """Format a ``time`` value as ``HH:MM AM/PM`` with zero-padded hour and minute."""

    hour = parsed.hour % 12 or 12
    ampm = "AM" if parsed.hour < 12 else "PM"
    if parsed.second or parsed.microsecond:
        return f"{hour:02d}:{parsed.minute:02d}:{parsed.second:02d} {ampm}"
    return f"{hour:02d}:{parsed.minute:02d} {ampm}"


def _format_metadata_date(value: date | str | None) -> str:
    """Return a normalized ``YYYY-MM-DD`` string, or empty when invalid."""

    if value is None:
        return ""
    if isinstance(value, date):
        return value.isoformat()
    parsed = _parse_metadata_date(str(value))
    return parsed.isoformat() if parsed else ""


def _format_metadata_time(value: time | str | None) -> str:
    """Return a normalized 12-hour AM/PM time string, or empty when invalid."""

    if value is None:
        return ""
    if isinstance(value, time):
        parsed = value
    else:
        parsed = _parse_metadata_time(str(value))
    if parsed is None:
        return ""
    return _format_metadata_time_display(parsed)


def _coerce_metadata_widget_state_value(field_number: int, raw_value: Any) -> Any:
    """Convert stored template or snapshot text to a widget-ready session value."""

    if field_number in DATE_METADATA_FIELD_NUMBERS:
        if isinstance(raw_value, date):
            return raw_value
        if raw_value is None:
            return None
        stripped = str(raw_value).strip()
        return _parse_metadata_date(stripped) if stripped else None
    if field_number in TIME_METADATA_FIELD_NUMBERS:
        if isinstance(raw_value, time):
            return _format_metadata_time_display(raw_value)
        if raw_value is None:
            return ""
        stripped = str(raw_value).strip()
        if not stripped:
            return ""
        normalized = _format_metadata_time(stripped)
        return normalized if normalized else _format_ampm_time_mask(stripped)
    if raw_value is None:
        return ""
    return str(raw_value).strip()


def _metadata_widget_state_to_string(field_number: int, raw_value: Any) -> str:
    """Serialize a metadata widget session value to workbook/template text."""

    if field_number in DATE_METADATA_FIELD_NUMBERS:
        return _format_metadata_date(raw_value)
    if field_number in TIME_METADATA_FIELD_NUMBERS:
        return _format_metadata_time(raw_value)
    if raw_value is None:
        return ""
    return str(raw_value).strip()


def _is_metadata_widget_value_empty(field_number: int, raw_value: Any) -> bool:
    """Return whether a metadata widget has no usable value."""

    return not _metadata_widget_state_to_string(field_number, raw_value)


def _collect_metadata_values() -> dict[str, str]:
    """Read metadata field values from Streamlit session state."""

    return {
        excel_label: _metadata_widget_state_to_string(
            field_number,
            st.session_state.get(widget_key),
        )
        for field_number, excel_label, widget_key in METADATA_FIELD_SPECS
    }


def _collect_metadata_values_by_widget_key(
    field_values: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Read metadata widget values from an explicit mapping or Streamlit session state.

    Parameters
    ----------
    field_values:
        Optional mapping of widget key to raw string. When omitted, session state is used.
    """

    if field_values is None:
        return {
            widget_key: _metadata_widget_state_to_string(
                field_number,
                st.session_state.get(widget_key),
            )
            for field_number, _, widget_key in METADATA_FIELD_SPECS
        }
    return {
        widget_key: _metadata_widget_state_to_string(field_number, field_values.get(widget_key))
        for field_number, _, widget_key in METADATA_FIELD_SPECS
    }


def _field_number_for_widget_key(widget_key: str) -> int | None:
    """Return the metadata field number for a widget key, if known."""

    for field_number, _, key in METADATA_FIELD_SPECS:
        if key == widget_key:
            return field_number
    return None


def _is_template_field_exportable(field_number: int, raw_value: str) -> bool:
    """
    Return whether a metadata field has a valid, exportable value.

    Numeric fields (1–6) must parse as numbers. Date (12) and Testing Time (13) must use
    valid ``YYYY-MM-DD`` and 12-hour AM/PM formats. Optional Notes (16) may be omitted
    when empty. All other fields must be non-empty text.
    """

    stripped = raw_value.strip() if isinstance(raw_value, str) else raw_value
    if field_number in OPTIONAL_METADATA_FIELD_NUMBERS:
        if _is_metadata_widget_value_empty(field_number, stripped):
            return False
        return True
    if field_number in NUMERIC_METADATA_FIELD_NUMBERS:
        return _parse_required_number(str(stripped).strip()) is not None
    if field_number in DATE_METADATA_FIELD_NUMBERS:
        return _format_metadata_date(stripped) != ""
    if field_number in TIME_METADATA_FIELD_NUMBERS:
        return _format_metadata_time(stripped) != ""
    return bool(str(stripped).strip())


def _serialize_template_field_value(
    field_number: int, raw_value: str | Any
) -> str | int | float | None:
    """Convert a raw widget value to a JSON-safe template value, or ``None`` when invalid."""

    if field_number in NUMERIC_METADATA_FIELD_NUMBERS:
        parsed = _parse_required_number(str(raw_value).strip())
        if parsed is None:
            return None
        return _to_excel_number(parsed)
    if field_number in DATE_METADATA_FIELD_NUMBERS:
        formatted = _format_metadata_date(raw_value)
        return formatted if formatted else None
    if field_number in TIME_METADATA_FIELD_NUMBERS:
        formatted = _format_metadata_time(raw_value)
        return formatted if formatted else None
    stripped = str(raw_value).strip()
    if not stripped:
        return None
    return stripped


def _build_template_export_payload(
    field_values: dict[str, str],
    selected_keys: frozenset[str],
) -> dict[str, Any] | None:
    """
    Build a setup-template JSON payload from selected metadata widget keys.

    Only checked fields with valid values are included.
    """

    export_fields: dict[str, str | int | float] = {}
    for field_number, _, widget_key in METADATA_FIELD_SPECS:
        if widget_key not in selected_keys:
            continue
        serialized = _serialize_template_field_value(field_number, field_values.get(widget_key, ""))
        if serialized is None:
            continue
        export_fields[widget_key] = serialized
    if not export_fields:
        return None
    return {"version": TEMPLATE_VERSION, "fields": export_fields}


def _apply_template_import_to_values(
    payload: dict[str, Any],
    field_values: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """
    Merge template payload values into a widget-key mapping.

    Returns
    -------
    tuple
        Updated values and human-readable warning messages.
    """

    warnings: list[str] = []
    if payload.get("version") != TEMPLATE_VERSION:
        warnings.append(f"Unsupported template version: {payload.get('version')!r}")
        return field_values, warnings

    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, dict):
        warnings.append("Template is missing a valid 'fields' object.")
        return field_values, warnings

    updated = dict(field_values)
    for widget_key, raw_value in raw_fields.items():
        if widget_key not in METADATA_WIDGET_KEYS:
            warnings.append(f"Unknown template field ignored: {widget_key}")
            continue
        field_number = _field_number_for_widget_key(widget_key)
        if field_number is None:
            continue
        if isinstance(raw_value, bool):
            warnings.append(f"Invalid value for {widget_key}; skipped.")
            continue
        if isinstance(raw_value, (int, float)):
            if field_number in NUMERIC_METADATA_FIELD_NUMBERS:
                updated[widget_key] = str(_to_excel_number(float(raw_value)))
            else:
                updated[widget_key] = str(raw_value)
            continue
        if isinstance(raw_value, str):
            if field_number in NUMERIC_METADATA_FIELD_NUMBERS:
                parsed = _parse_required_number(raw_value)
                if parsed is None:
                    warnings.append(f"Invalid numeric value for {widget_key}; skipped.")
                    continue
                updated[widget_key] = str(_to_excel_number(parsed))
            elif field_number in DATE_METADATA_FIELD_NUMBERS:
                formatted = _format_metadata_date(raw_value)
                if not formatted:
                    warnings.append(f"Invalid date value for {widget_key}; skipped.")
                    continue
                updated[widget_key] = formatted
            elif field_number in TIME_METADATA_FIELD_NUMBERS:
                formatted = _format_metadata_time(raw_value)
                if not formatted:
                    warnings.append(f"Invalid time value for {widget_key}; skipped.")
                    continue
                updated[widget_key] = formatted
            else:
                updated[widget_key] = raw_value.strip()
            continue
        warnings.append(f"Unsupported value type for {widget_key}; skipped.")

    return updated, warnings


def _template_selection_key(widget_key: str) -> str:
    """Session-state key for a setup-template export checkbox."""

    return f"txt2excel_template_sel_{widget_key}"


def _all_template_export_fields_selected() -> bool:
    """Return whether every metadata export checkbox is selected."""

    return all(
        st.session_state.get(_template_selection_key(widget_key), False)
        for _, _, widget_key in METADATA_FIELD_SPECS
    )


def _apply_template_export_select_all() -> None:
    """Mirror the master select-all checkbox to every export field."""

    select_all = st.session_state.get(TEMPLATE_EXPORT_SELECT_ALL_KEY, False)
    for _, _, widget_key in METADATA_FIELD_SPECS:
        st.session_state[_template_selection_key(widget_key)] = select_all


def _validate_metadata() -> str | None:
    """Return an error message when any required metadata field is empty or invalid."""

    missing: list[str] = []
    invalid: list[str] = []
    for field_number, label, widget_key in METADATA_FIELD_SPECS:
        if field_number in OPTIONAL_METADATA_FIELD_NUMBERS:
            continue
        raw_value = st.session_state.get(widget_key)
        if _is_metadata_widget_value_empty(field_number, raw_value):
            missing.append(label)
            continue
        if field_number in DATE_METADATA_FIELD_NUMBERS and not _format_metadata_date(raw_value):
            invalid.append(f"{field_number}. {label} must be a valid date (YYYY-MM-DD).")
        elif field_number in TIME_METADATA_FIELD_NUMBERS and not _format_metadata_time(raw_value):
            invalid.append(f"{field_number}. {label} must be a valid time (e.g. 01:30 PM).")
    if missing:
        return f"Required metadata fields are missing: {', '.join(missing)}"
    if invalid:
        return invalid[0]
    return None


def _validate_prep_inputs(
    sorted_files: list[Any],
    target_inputs: list[str],
    actual_inputs: list[str],
) -> tuple[dict[str, Any], list[int | float], list[int | float], str | None]:
    """
    Validate all user inputs before workbook export.

    Returns
    -------
    tuple
        ``(metadata, target_concentrations, actual_concentrations, error_message)``.
    """

    metadata_error = _validate_metadata()
    if metadata_error:
        return {}, [], [], metadata_error

    metadata: dict[str, Any] = {}
    for field_number, excel_label, widget_key in METADATA_FIELD_SPECS:
        raw_value = st.session_state.get(widget_key)
        if field_number in NUMERIC_METADATA_FIELD_NUMBERS:
            parsed = _parse_required_number(
                _metadata_widget_state_to_string(field_number, raw_value)
            )
            if parsed is None:
                return (
                    {},
                    [],
                    [],
                    f"{field_number}. {excel_label} must be a number.",
                )
            metadata[excel_label] = _to_excel_number(parsed)
        elif field_number in DATE_METADATA_FIELD_NUMBERS:
            formatted = _format_metadata_date(raw_value)
            if not formatted:
                return (
                    {},
                    [],
                    [],
                    f"{field_number}. {excel_label} must be a valid date (YYYY-MM-DD).",
                )
            metadata[excel_label] = formatted
        elif field_number in TIME_METADATA_FIELD_NUMBERS:
            formatted = _format_metadata_time(raw_value)
            if not formatted:
                return (
                    {},
                    [],
                    [],
                    f"{field_number}. {excel_label} must be a valid time (e.g. 01:30 PM).",
                )
            metadata[excel_label] = formatted
        else:
            metadata[excel_label] = _metadata_widget_state_to_string(field_number, raw_value)

    target_values: list[int | float] = []
    for file, raw_target in zip(sorted_files, target_inputs, strict=True):
        parsed_target = _parse_required_number(raw_target)
        if parsed_target is None:
            return (
                {},
                [],
                [],
                (
                    f"{PREP_TARGET_CONCENTRATION_HEADER} for **{file.name}** must be a number. "
                    "Use 0 for rinsate-only controls."
                ),
            )
        target_values.append(_to_excel_number(parsed_target))

    actual_values: list[int | float] = []
    for file, raw_actual in zip(sorted_files, actual_inputs, strict=True):
        parsed_actual = _parse_required_number(raw_actual)
        if parsed_actual is None:
            return (
                {},
                [],
                [],
                f"{PREP_ACTUAL_CONCENTRATION_HEADER} for **{file.name}** must be a number.",
            )
        actual_values.append(_to_excel_number(parsed_actual))

    return metadata, target_values, actual_values, None


def _build_embedded_workbook_rows(
    metadata: dict[str, Any],
    *,
    target_concentrations: list[int | float],
    actual_concentrations: list[int | float],
    raman_shift: np.ndarray,
    intensity_columns: list[np.ndarray],
) -> list[list[Any]]:
    """
    Build row-major cell data for the embedded-metadata Excel layout.

    Parameters
    ----------
    metadata:
        Mapping of Excel column-A labels to column-B values (rows 1–16).
    target_concentrations:
        Target concentration labels per signal column.
    actual_concentrations:
        Measured concentration values per signal column.
    raman_shift:
        Raman shift grid (cm⁻¹).
    intensity_columns:
        One intensity vector per signal column.

    Returns
    -------
    list[list[Any]]
        Rectangular rows suitable for ``DataFrame.to_excel(header=False)``.
    """

    n_signals = len(intensity_columns)
    width = 1 + n_signals
    rows: list[list[Any]] = []

    for _, excel_label, _ in METADATA_FIELD_SPECS:
        row = [excel_label, metadata[excel_label]]
        row.extend([""] * (width - len(row)))
        rows.append(row)

    target_row = [TARGET_CONCENTRATION_LABEL, *target_concentrations]
    target_row.extend([""] * (width - len(target_row)))
    rows.append(target_row)

    actual_row = [ACTUAL_CONCENTRATION_LABEL, *actual_concentrations]
    actual_row.extend([""] * (width - len(actual_row)))
    rows.append(actual_row)

    header_row = [RAMAN_SHIFT_HEADER, *[INTENSITY_HEADER] * n_signals]
    rows.append(header_row)

    for shift_idx, shift_value in enumerate(raman_shift):
        data_row = [float(shift_value)]
        for signal_idx in range(n_signals):
            data_row.append(float(intensity_columns[signal_idx][shift_idx]))
        rows.append(data_row)

    return rows


# Metadata field numbers (Excel row index) after which a full-width separator line is drawn.
_SEPARATOR_AFTER_METADATA_NUMBERS = frozenset({4, 6, 14})


def _apply_full_width_row_separator(
    worksheet: Any,
    row_idx: int,
    max_col: int,
    bottom_side: Any,
) -> None:
    """
    Draw a horizontal separator beneath ``row_idx`` across columns 1..``max_col``.

    Only the bottom edge is styled; no left, right, or top borders are added.
    """
    from openpyxl.styles import Border

    for col_idx in range(1, max_col + 1):
        worksheet.cell(row=row_idx, column=col_idx).border = Border(bottom=bottom_side)


def _embedded_workbook_to_excel_bytes(rows: list[list[Any]]) -> bytes:
    """
    Serialize embedded workbook rows to styled ``.xlsx`` bytes.

    Applies bold labels and full-width horizontal separators between metadata
    sections, before the spectral table, and after the actual concentration row.
    """

    from openpyxl import Workbook
    from openpyxl.styles import Font, Side

    n_metadata = len(METADATA_FIELD_SPECS)
    n_signals = len(rows[n_metadata]) - 1
    data_width = 1 + n_signals
    target_row_idx = n_metadata + 1
    actual_row_idx = n_metadata + 2
    header_row_idx = n_metadata + 3

    separator_row_indices = [
        field_number
        for field_number, _, _ in METADATA_FIELD_SPECS
        if field_number in _SEPARATOR_AFTER_METADATA_NUMBERS
    ]
    separator_row_indices.append(actual_row_idx)

    thin_side = Side(style="thin", color="000000")
    bold_font = Font(bold=True)

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"

    for row_idx, row_values in enumerate(rows, start=1):
        for col_idx, value in enumerate(row_values, start=1):
            if value == "":
                continue
            worksheet.cell(row=row_idx, column=col_idx, value=value)

        if row_idx <= n_metadata:
            worksheet.cell(row=row_idx, column=1).font = bold_font
            continue

        if row_idx in {target_row_idx, actual_row_idx}:
            worksheet.cell(row=row_idx, column=1).font = bold_font
            continue

        if row_idx == header_row_idx:
            for col_idx in range(1, data_width + 1):
                worksheet.cell(row=row_idx, column=col_idx).font = bold_font

    for separator_row_idx in separator_row_indices:
        _apply_full_width_row_separator(worksheet, separator_row_idx, data_width, thin_side)

    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _get_merged_preview() -> dict[str, Any] | None:
    """
    Return a valid merged-spectrum preview payload from session state.

    Clears stale values from older app versions (e.g. a legacy DataFrame).
    """

    payload = st.session_state.get(MERGED_PREVIEW_KEY)
    if payload is None:
        return None
    if isinstance(payload, pd.DataFrame) or not isinstance(payload, dict):
        st.session_state.pop(MERGED_PREVIEW_KEY, None)
        return None
    required_keys = {"raman_shift", "intensity_columns", "labels"}
    if not required_keys.issubset(payload.keys()):
        st.session_state.pop(MERGED_PREVIEW_KEY, None)
        return None
    return payload


def _process_uploaded_template(
    uploaded_template: Any,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    """
    Parse a setup-template JSON upload and compute merged widget values.

    Returns
    -------
    tuple
        Feedback with ``level``, ``message``, and ``warnings``, plus merged widget
        values when parsing succeeds. Widget session state is not modified here;
        callers must defer application until before metadata widgets render.
    """

    try:
        payload = json.loads(uploaded_template.getvalue().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "level": "error",
            "message": f"Could not read template file: {exc}",
            "warnings": [],
        }, None

    if not isinstance(payload, dict):
        return {
            "level": "error",
            "message": "Template file must contain a JSON object.",
            "warnings": [],
        }, None

    before_values = _collect_metadata_values_by_widget_key()
    updated_values, warnings = _apply_template_import_to_values(payload, before_values)
    applied_count = sum(
        1
        for widget_key, value in updated_values.items()
        if value != before_values.get(widget_key, "")
    )
    if applied_count:
        message = f"Loaded template values for {applied_count} field(s)."
        level = "success"
    else:
        message = "Template applied; metadata already matched the file."
        level = "info"
    return {"level": level, "message": message, "warnings": warnings}, updated_values


def _render_template_import_feedback(container: DeltaGenerator) -> None:
    """Show import feedback stored from the prior apply-and-reset cycle."""

    feedback = st.session_state.pop(TEMPLATE_IMPORT_FEEDBACK_KEY, None)
    if not isinstance(feedback, dict):
        return
    level = feedback.get("level")
    message = feedback.get("message", "")
    warnings = feedback.get("warnings", [])
    if level == "success" and message:
        container.success(message)
    elif level == "info" and message:
        container.info(message)
    elif level == "error" and message:
        container.error(message)
    for warning in warnings:
        if warning:
            container.warning(warning)


def _render_export_template(container: DeltaGenerator) -> None:
    """Render selective export controls in a popover matching the import layout."""

    field_values = _collect_metadata_values_by_widget_key()
    with container.popover("Export metadata to template", use_container_width=True):
        st.caption(
            "Save metadata defaults to a JSON file. Only checked fields with valid "
            "values are included — you do not need to complete the form."
        )
        st.caption("Select fields to include in the download.")
        for field_number, _, widget_key in METADATA_FIELD_SPECS:
            selection_key = _template_selection_key(widget_key)
            if selection_key not in st.session_state:
                st.session_state[selection_key] = _is_template_field_exportable(
                    field_number,
                    field_values.get(widget_key, ""),
                )
        st.session_state[TEMPLATE_EXPORT_SELECT_ALL_KEY] = _all_template_export_fields_selected()
        st.checkbox(
            "Select all",
            key=TEMPLATE_EXPORT_SELECT_ALL_KEY,
            on_change=_apply_template_export_select_all,
        )
        n_columns = 2
        for row_start in range(0, len(METADATA_FIELD_SPECS), n_columns):
            row_specs = METADATA_FIELD_SPECS[row_start : row_start + n_columns]
            columns = st.columns(n_columns)
            for column, (field_number, excel_label, widget_key) in zip(
                columns, row_specs, strict=False
            ):
                selection_key = _template_selection_key(widget_key)
                column.checkbox(
                    f"{field_number}. {excel_label}",
                    key=selection_key,
                )

        selected_keys = frozenset(
            widget_key
            for _, _, widget_key in METADATA_FIELD_SPECS
            if st.session_state.get(_template_selection_key(widget_key), False)
        )
        export_payload = _build_template_export_payload(field_values, selected_keys)
        if export_payload is None:
            st.button(
                "Export metadata to template",
                disabled=True,
                help="Select at least one field with a valid value to export.",
                key="txt2excel_template_export_disabled",
                use_container_width=True,
            )
        else:
            st.download_button(
                "Export metadata to template",
                data=json.dumps(export_payload, indent=2),
                file_name=TEMPLATE_EXPORT_FILENAME,
                mime="application/json",
                key="txt2excel_template_export",
                use_container_width=True,
            )


def _render_import_template(container: DeltaGenerator) -> None:
    """
    Render import as a popover button matching the export template layout.

    ``st.file_uploader`` lives inside the popover because Streamlit does not
    expose a file-picker API on ``st.button``.
    """

    uploader_key = (
        f"txt2excel_template_import_"
        f"{st.session_state.get(TEMPLATE_IMPORT_UPLOADER_RESET_KEY, 'default')}"
    )
    with container.popover("Import metadata from template", use_container_width=True):
        st.caption(
            "Load a previously saved JSON file into the metadata form above. "
            "Every field stored in that file is applied — export field selection does not "
            "affect import."
        )
        st.caption("Select a `.json` setup file. Values apply as soon as the file is chosen.")
        uploaded_template = st.file_uploader(
            "JSON setup file",
            type=["json"],
            label_visibility="collapsed",
            key=uploader_key,
        )
        if uploaded_template is not None:
            feedback, updated_values = _process_uploaded_template(uploaded_template)
            st.session_state[TEMPLATE_IMPORT_FEEDBACK_KEY] = feedback
            if updated_values is not None:
                st.session_state[TEMPLATE_IMPORT_PENDING_VALUES_KEY] = updated_values
            st.session_state[TEMPLATE_IMPORT_UPLOADER_RESET_KEY] = str(uuid.uuid4())
            st.rerun()
    _render_template_import_feedback(container)


def _apply_testing_time_mask() -> None:
    """Reformat the testing-time text input as 12-hour AM/PM from typed digits."""

    st.session_state[TESTING_TIME_WIDGET_KEY] = _format_ampm_time_mask(
        st.session_state.get(TESTING_TIME_WIDGET_KEY, "")
    )


def _prepare_testing_time_widget_state(widget_key: str) -> None:
    """Normalize testing-time session state before rendering the masked input."""

    if widget_key not in st.session_state:
        st.session_state[widget_key] = ""
        return
    raw_value = st.session_state[widget_key]
    if isinstance(raw_value, time):
        st.session_state[widget_key] = _format_metadata_time_display(raw_value)
        return
    if raw_value is None:
        st.session_state[widget_key] = ""
        return
    text_value = str(raw_value).strip()
    if not text_value:
        st.session_state[widget_key] = ""
        return
    normalized = _format_metadata_time(text_value)
    st.session_state[widget_key] = normalized or _format_ampm_time_mask(text_value)


def _render_metadata_field_widget(
    column: DeltaGenerator,
    field_number: int,
    excel_label: str,
    widget_key: str,
) -> None:
    """Render a single metadata widget using the appropriate Streamlit input type."""

    label = f"{field_number}. {excel_label}"
    if field_number in DATE_METADATA_FIELD_NUMBERS:
        if widget_key not in st.session_state:
            st.session_state[widget_key] = None
        column.date_input(label, key=widget_key, format="YYYY-MM-DD")
        return
    if field_number in TIME_METADATA_FIELD_NUMBERS:
        _prepare_testing_time_widget_state(widget_key)
        column.text_input(
            label,
            key=widget_key,
            placeholder="hh:mm AM/PM",
            on_change=_apply_testing_time_mask,
        )
        return
    column.text_input(label, key=widget_key)


def _render_metadata_fields(container: DeltaGenerator) -> None:
    """Render numbered metadata inputs in four columns, filled row by row."""

    container.markdown("### Experiment metadata")
    if st.session_state.pop(RESTORE_METADATA_AFTER_RELOAD_KEY, False):
        _restore_persistent_metadata_widgets()
    _apply_pending_template_import_values()
    n_columns = 4
    for row_start in range(0, len(METADATA_FIELD_SPECS), n_columns):
        row_specs = METADATA_FIELD_SPECS[row_start : row_start + n_columns]
        columns = container.columns(n_columns)
        for column, (number, excel_label, widget_key) in zip(columns, row_specs, strict=False):
            _render_metadata_field_widget(column, number, excel_label, widget_key)
    _sync_persistent_metadata_snapshot()

    export_col, import_col = container.columns(2)
    _render_export_template(export_col)
    _render_import_template(import_col)


def _read_shift_bounds() -> tuple[str, str]:
    """Read min/max Raman shift widget values from session state."""

    min_shift = str(st.session_state.get("txt2excel_min_shift", DEFAULT_MIN_SHIFT)).strip()
    max_shift = str(st.session_state.get("txt2excel_max_shift", "")).strip()
    return min_shift, max_shift


def _sync_shift_defaults_for_upload(
    uploaded_files: list[Any],
    file_contents: dict[str, str],
) -> tuple[np.ndarray | None, str | None]:
    """
    Validate Raman shift grids across uploads and refresh default max-shift bounds.

    Returns
    -------
    tuple
        ``(common_shift, shift_error)``.
    """

    common_shift, shift_error = _validate_common_shift(file_contents)
    if shift_error or not uploaded_files:
        return common_shift, shift_error

    default_max_shift = str(common_shift.max()) if common_shift is not None else ""
    file_signature = tuple(sorted(file.name for file in uploaded_files))
    if st.session_state.get("txt2excel_file_signature") != file_signature:
        st.session_state["txt2excel_file_signature"] = file_signature
        if default_max_shift:
            st.session_state["txt2excel_max_shift"] = default_max_shift

    if "txt2excel_min_shift" not in st.session_state:
        st.session_state["txt2excel_min_shift"] = str(DEFAULT_MIN_SHIFT)

    return common_shift, shift_error


def _render_raman_shift_bounds(container: DeltaGenerator) -> None:
    """Render the Raman shift range subsection with min/max inputs."""

    container.markdown("### Raman shift range")
    col_min, col_max = container.columns(2)
    with col_min:
        col_min.text_input(
            "Min Raman Shift",
            key="txt2excel_min_shift",
        )
    with col_max:
        col_max.text_input(
            "Max Raman Shift",
            key="txt2excel_max_shift",
        )


def render_prep_mode() -> None:
    """Render the full TXT-to-Excel prep utility (separate app shell)."""

    with st.sidebar:
        st.button(
            "← Back to Analysis",
            key="exit_txt_to_excel_mode",
            on_click=enter_analysis_mode,
            use_container_width=True,
        )
        st.markdown("---")
        header_col, btn_col = st.columns([3, 1])
        with header_col:
            st.markdown("# 📁 Upload Raman TXT Files")
        with btn_col:
            st.button("Reload Data", type="primary", on_click=clear_prep_uploads)
        uploaded_files = st.file_uploader(
            "Upload Raman TXT Files",
            type=["txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"txt_to_excel_uploader_{st.session_state.get(PREP_UPLOADER_RESET_KEY, 'default')}",
        )

    file_contents: dict[str, str] = {}
    if uploaded_files:
        for file in uploaded_files:
            file_contents[file.name] = file.read().decode("utf-8")

    shift_error: str | None = None
    if uploaded_files:
        _, shift_error = _sync_shift_defaults_for_upload(uploaded_files, file_contents)
        if shift_error:
            with st.sidebar:
                st.error(f"❌ {shift_error}")

    st.title("Raman TXT to Excel Merger")
    st.caption(
        "Enter experiment metadata, map concentrations to each uploaded spectrum, "
        "then download the embedded workbook."
    )

    if not uploaded_files:
        st.info(
            "Upload one or more `.txt` files using the sidebar, then complete metadata "
            "fields 1–15 (field 16 Notes optional) and concentrations below."
        )
        st.markdown("### Next steps")
        st.markdown(
            "1. Upload `.txt` files in the sidebar.\n"
            "2. Complete metadata fields 1–15.\n"
            "3. Set the Raman shift range and enter target and actual concentrations for each file.\n"
            "4. Convert and download the embedded `.xlsx`.\n"
            "5. Switch back to **Analysis** and upload the saved file."
        )
        return

    if shift_error:
        st.warning("Fix Raman shift mismatches before continuing.")
        return

    _render_metadata_fields(st)
    st.markdown("---")

    sorted_files = sorted(uploaded_files, key=lambda file: _extract_cfu_sort_key(file.name))
    n_files = len(sorted_files)

    _render_raman_shift_bounds(st)

    st.markdown("### Concentrations")
    st.caption(
        f"{n_files} signal column{'s' if n_files != 1 else ''} will be written. "
        "Target and actual concentrations must be numbers. Use **0** for rinsate-only controls."
    )

    header_file, header_target, header_actual = st.columns([2, 1, 1])
    with header_file:
        st.markdown("**File**")
    with header_target:
        st.markdown(f"**{PREP_TARGET_CONCENTRATION_HEADER}**")
    with header_actual:
        st.markdown(f"**{PREP_ACTUAL_CONCENTRATION_HEADER}**")

    target_inputs: list[str] = []
    actual_inputs: list[str] = []
    for file in sorted_files:
        col_file, col_target, col_actual = st.columns([2, 1, 1])
        with col_file:
            st.markdown(
                f"<div style='padding-top:6px'>{file.name}</div>",
                unsafe_allow_html=True,
            )
        with col_target:
            target_inputs.append(
                st.text_input(
                    label=f"Target for {file.name}",
                    key=f"txt2excel_target_{file.name}",
                    label_visibility="collapsed",
                    placeholder="e.g. 1000",
                )
            )
        with col_actual:
            actual_inputs.append(
                st.text_input(
                    label=f"Actual for {file.name}",
                    key=f"txt2excel_actual_{file.name}",
                    label_visibility="collapsed",
                    placeholder="e.g. 995",
                )
            )

    st.markdown("---")
    col_name_input, col_ext = st.columns([9, 1])
    with col_name_input:
        output_file_name = st.text_input(
            "Output File Name",
            placeholder="Enter output file name",
            key="txt2excel_output_name",
        )
    with col_ext:
        st.markdown(
            "<div style='padding-top:38px;font-weight:600'>.xlsx</div>",
            unsafe_allow_html=True,
        )

    convert_clicked = st.button(
        "Convert and Export",
        key="txt2excel_convert_button",
        type="primary",
    )

    if convert_clicked:
        if not output_file_name.strip():
            st.warning("Please enter an output file name.")
            return

        metadata, target_values, actual_values, validation_error = _validate_prep_inputs(
            sorted_files,
            target_inputs,
            actual_inputs,
        )
        if validation_error:
            st.error(validation_error)
            return

        min_shift_input, max_shift_input = _read_shift_bounds()
        try:
            min_shift = float(min_shift_input) if min_shift_input else None
            max_shift = float(max_shift_input) if max_shift_input else None
        except ValueError:
            st.error("Raman shift values must be numeric.")
            return

        ordered_contents = {file.name: file_contents[file.name] for file in sorted_files}
        raman_shift, intensity_columns, merge_error = _merge_txt_spectra(
            ordered_contents,
            min_shift=min_shift,
            max_shift=max_shift,
        )
        if merge_error or raman_shift is None or intensity_columns is None:
            st.error(f"❌ {merge_error or 'Merge failed.'}")
            st.session_state.pop(MERGED_PREVIEW_KEY, None)
            st.session_state.pop(EXPORT_BYTES_KEY, None)
            st.session_state.pop(EXPORT_FILENAME_KEY, None)
            return

        workbook_rows = _build_embedded_workbook_rows(
            metadata,
            target_concentrations=target_values,
            actual_concentrations=actual_values,
            raman_shift=raman_shift,
            intensity_columns=intensity_columns,
        )
        excel_bytes = _embedded_workbook_to_excel_bytes(workbook_rows)
        export_filename = f"{output_file_name.strip()}.xlsx"

        st.session_state[MERGED_PREVIEW_KEY] = {
            "raman_shift": raman_shift,
            "intensity_columns": intensity_columns,
            "labels": [
                f"Target {target} / Actual {actual}"
                for target, actual in zip(target_values, actual_values, strict=True)
            ],
        }
        st.session_state[EXPORT_BYTES_KEY] = excel_bytes
        st.session_state[EXPORT_FILENAME_KEY] = export_filename
        logger.info(
            "Embedded TXT merge complete: %d files → %d Raman rows, %d signal columns",
            len(sorted_files),
            len(raman_shift),
            len(intensity_columns),
        )

    export_bytes = st.session_state.get(EXPORT_BYTES_KEY)
    export_filename = st.session_state.get(EXPORT_FILENAME_KEY)
    if export_bytes and export_filename:
        st.success("Merge complete. Download the embedded workbook below.")
        st.download_button(
            "Download Embedded Excel",
            data=export_bytes,
            file_name=str(export_filename),
            key="txt2excel_download_button",
            type="primary",
        )

    preview_payload = _get_merged_preview()
    if preview_payload is not None:
        st.markdown("---")
        st.markdown("### Preview of Merged Signals")
        fig, ax = plt.subplots(figsize=(10, 6))
        preview_shift = preview_payload["raman_shift"]
        for intensities, label in zip(
            preview_payload["intensity_columns"],
            preview_payload["labels"],
            strict=True,
        ):
            ax.plot(preview_shift, intensities, label=label)
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel(INTENSITY_HEADER)
        ax.set_title("Merged Signal Preview")
        ax.grid(True)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
