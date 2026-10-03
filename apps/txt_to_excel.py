"""
Instrument TXT to Excel merger — Streamlit prep utility.

Converts raw Raman instrument ``.txt`` exports into embedded-metadata Excel
workbooks using the independently callable data converter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
import unicodedata
from html import escape
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import streamlit as st

from components.shared_ui import render_figure_stretch
from sensd_sers_analysis.config.metadata_schema import (
    DATE_METADATA_FIELD_NUMBERS,
    METADATA_FIELD_SPECS,
    METADATA_LOGICAL_GROUPS,
    METADATA_WIDGET_KEYS,
    NUMERIC_METADATA_FIELD_NUMBERS,
    OPTIONAL_METADATA_FIELD_NUMBERS,
    TIME_METADATA_FIELD_NUMBERS,
    PER_SIGNAL_PRESET_KEY,
    SAMPLE_TYPE_OPTIONS,
    SPECIAL_TREATMENT_OPTIONS,
)
from sensd_sers_analysis.data.txt_converter import (
    TEMPLATE_VERSION,
    apply_template_import_to_values,
    build_embedded_workbook_rows,
    build_template_export_payload,
    coerce_metadata_widget_state_value,
    embedded_workbook_to_excel_bytes,
    field_number_for_widget_key,
    format_metadata_date,
    format_metadata_time,
    is_metadata_widget_value_empty,
    is_template_field_exportable,
    merge_txt_spectra,
    metadata_widget_state_to_string,
    parse_optional_number,
    parse_required_number,
    to_excel_number,
    validate_common_shift,
)
from sensd_sers_analysis.application.metadata_presets import validate_signal_preset
from sensd_sers_analysis.visualization import plot_spectra

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

logger = logging.getLogger(__name__)

APP_MODE_KEY = "_app_mode"
APP_MODE_ANALYSIS = "analysis"
APP_MODE_PREP = "prep"

MERGED_PREVIEW_KEY = "_txt2excel_merged_preview"
EXPORT_BYTES_KEY = "_txt2excel_export_bytes"
EXPORT_FILENAME_KEY = "_txt2excel_export_filename"
EXPORT_CONTEXT_KEY = "_txt2excel_export_context"
PREP_UPLOADER_RESET_KEY = "_txt2excel_uploader_reset"
TEMPLATE_IMPORT_UPLOADER_RESET_KEY = "_txt2excel_template_import_reset"
TEMPLATE_IMPORT_FEEDBACK_KEY = "_txt2excel_template_import_feedback"
TEMPLATE_IMPORT_PENDING_VALUES_KEY = "_txt2excel_template_import_pending_values"
TEMPLATE_EXPORT_SELECT_ALL_KEY = "txt2excel_template_export_select_all"
TEMPLATE_EXPORT_FILENAME = "SERS_metadata_preset.json"
DEFAULT_MIN_SHIFT = 560.9
MERGED_PREVIEW_FIGSIZE = (10.0, 6.0)
MERGED_PREVIEW_HUE_COL = "concentration_cfu_ml"
MERGED_PREVIEW_LEGEND_TITLE = "Concentrations (CFU/mL)"

PREP_TARGET_CONCENTRATION_HEADER = "Initial Target Concentration"
PREP_ACTUAL_CONCENTRATION_HEADER = "Actual Concentration"
PREP_SPECIAL_TREATMENT_HEADER = "Special Treatment"
REQUIRED_FIELD_LABEL_SUFFIX = " *"
OPTIONAL_FIELD_LABEL_SUFFIX = " (optional)"

METADATA_UI_COLUMN_COUNT = 5
# Three five-column UI rows (15 fields) before full-width notes.
METADATA_UI_ROWS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4, 5),
    (6, 7, 8, 9, 10),
    (11, 12, 13, 14, 15),
)
_PREP_LAYOUT_STABILITY_CSS_KEY = "_txt2excel_prep_layout_css_injected"
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
    st.session_state.pop(EXPORT_CONTEXT_KEY, None)


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
            st.session_state[key] = coerce_metadata_widget_state_value(field_number, snapshot[key])


def _apply_pending_template_import_values() -> None:
    """Apply deferred template-import values before metadata widgets are rendered."""

    pending_values = st.session_state.pop(TEMPLATE_IMPORT_PENDING_VALUES_KEY, None)
    if not isinstance(pending_values, dict):
        return
    if PER_SIGNAL_PRESET_KEY in pending_values:
        preset = pending_values.pop(PER_SIGNAL_PRESET_KEY)
        st.session_state[PER_SIGNAL_PRESET_KEY] = preset
        _clear_session_keys_by_prefix("txt2excel_sample_type_")
        _clear_session_keys_by_prefix("txt2excel_treatment_")
    for widget_key, value in pending_values.items():
        if widget_key not in METADATA_WIDGET_KEYS:
            continue
        field_number = field_number_for_widget_key(widget_key)
        if field_number is None:
            continue
        st.session_state[widget_key] = coerce_metadata_widget_state_value(field_number, value)
    _sync_persistent_metadata_snapshot(
        {
            widget_key: metadata_widget_state_to_string(
                field_number, st.session_state.get(widget_key)
            )
            for widget_key in pending_values
            if widget_key in METADATA_WIDGET_KEYS
            for field_number in [field_number_for_widget_key(widget_key)]
            if field_number is not None
        }
    )


def clear_prep_uploads() -> None:
    """
    Reset uploaded TXT files and derived export artifacts.

    Instrument metadata (fields 1–7), date, operator, and rinsate type persist;
    run-specific fields, per-signal labels, concentrations, and Raman bounds reset.
    Explicitly imported filename-keyed label presets remain available.
    """

    logger.info("Clearing prep upload state (Reload Data clicked)")
    widget_values = {
        widget_key: metadata_widget_state_to_string(
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
    st.session_state.pop(EXPORT_CONTEXT_KEY, None)
    st.session_state.pop("txt2excel_file_signature", None)
    st.session_state.pop("txt2excel_min_shift", None)
    st.session_state.pop("txt2excel_max_shift", None)
    for widget_key in RELOAD_CLEAR_WIDGET_KEYS:
        field_number = field_number_for_widget_key(widget_key)
        if field_number in DATE_METADATA_FIELD_NUMBERS | TIME_METADATA_FIELD_NUMBERS:
            st.session_state[widget_key] = None
        else:
            st.session_state[widget_key] = ""
    _clear_session_keys_by_prefix("txt2excel_target_")
    _clear_session_keys_by_prefix("txt2excel_actual_")
    _clear_session_keys_by_prefix("txt2excel_treatment_")
    _clear_session_keys_by_prefix("txt2excel_sample_type_")
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


def _collect_metadata_values() -> dict[str, str]:
    """Read metadata field values from Streamlit session state."""

    return {
        excel_label: metadata_widget_state_to_string(
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
            widget_key: metadata_widget_state_to_string(
                field_number,
                st.session_state.get(widget_key),
            )
            for field_number, _, widget_key in METADATA_FIELD_SPECS
        }
    return {
        widget_key: metadata_widget_state_to_string(field_number, field_values.get(widget_key))
        for field_number, _, widget_key in METADATA_FIELD_SPECS
    }


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
        if is_metadata_widget_value_empty(field_number, raw_value):
            missing.append(label)
            continue
        if field_number in DATE_METADATA_FIELD_NUMBERS and not format_metadata_date(raw_value):
            invalid.append(f"{field_number}. {label} must be a valid date (YYYY-MM-DD).")
        elif field_number in TIME_METADATA_FIELD_NUMBERS and not format_metadata_time(raw_value):
            invalid.append(f"{field_number}. {label} must be a valid time (e.g. 01:30 PM).")
    if missing:
        return f"Required metadata fields are missing: {', '.join(missing)}"
    if invalid:
        return invalid[0]
    return None


def _prep_field_label(text: str, *, optional: bool = False) -> str:
    """Format a prep UI label with required (``*``) or optional suffix."""

    suffix = OPTIONAL_FIELD_LABEL_SUFFIX if optional else REQUIRED_FIELD_LABEL_SUFFIX
    return f"{text}{suffix}"


def _prep_column_header_markup(text: str, *, optional: bool = False) -> str:
    """
    Format a column header for ``st.markdown`` with bold text.

    The required asterisk or optional suffix is placed outside the bold span so
    markdown parsing does not treat ``*`` as emphasis delimiters.
    """

    suffix = OPTIONAL_FIELD_LABEL_SUFFIX if optional else REQUIRED_FIELD_LABEL_SUFFIX
    return f"**{text}**{suffix}"


def _metadata_field_label(field_number: int, excel_label: str) -> str:
    """Format a numbered metadata widget label with required/optional suffix."""

    base = f"{field_number}. {excel_label}"
    if field_number in OPTIONAL_METADATA_FIELD_NUMBERS:
        return _prep_field_label(base, optional=True)
    return _prep_field_label(base, optional=False)


def _validate_prep_inputs(
    sorted_files: list[Any],
    target_inputs: list[str],
    actual_inputs: list[str],
    *,
    special_treatment_inputs: list[str] | None = None,
) -> tuple[
    dict[str, Any],
    list[int | float | str],
    list[int | float],
    list[str],
    list[str],
    str | None,
]:
    """
    Validate all user inputs before workbook export.

    Returns
    -------
    tuple
        ``(metadata, target_concentrations, actual_concentrations, source_txt_filenames,
        special_treatments, error_message)``.
    """

    metadata_error = _validate_metadata()
    if metadata_error:
        return {}, [], [], [], [], metadata_error

    metadata: dict[str, Any] = {}
    for field_number, excel_label, widget_key in METADATA_FIELD_SPECS:
        raw_value = st.session_state.get(widget_key)
        if field_number in NUMERIC_METADATA_FIELD_NUMBERS:
            parsed = parse_required_number(metadata_widget_state_to_string(field_number, raw_value))
            if parsed is None:
                return (
                    {},
                    [],
                    [],
                    [],
                    [],
                    f"{field_number}. {excel_label} must be a number.",
                )
            metadata[excel_label] = to_excel_number(parsed)
        elif field_number in DATE_METADATA_FIELD_NUMBERS:
            formatted = format_metadata_date(raw_value)
            if not formatted:
                return (
                    {},
                    [],
                    [],
                    [],
                    [],
                    f"{field_number}. {excel_label} must be a valid date (YYYY-MM-DD).",
                )
            metadata[excel_label] = formatted
        elif field_number in TIME_METADATA_FIELD_NUMBERS:
            formatted = format_metadata_time(raw_value)
            if not formatted:
                return (
                    {},
                    [],
                    [],
                    [],
                    [],
                    f"{field_number}. {excel_label} must be a valid time (e.g. 01:30 PM).",
                )
            metadata[excel_label] = formatted
        else:
            metadata[excel_label] = metadata_widget_state_to_string(field_number, raw_value)

    target_values: list[int | float | str] = []
    for file, raw_target in zip(sorted_files, target_inputs, strict=True):
        stripped_target = raw_target.strip()
        if not stripped_target:
            target_values.append("")
            continue
        parsed_target = parse_optional_number(raw_target)
        if parsed_target is None or parsed_target < 0:
            return (
                {},
                [],
                [],
                [],
                [],
                (
                    f"{PREP_TARGET_CONCENTRATION_HEADER} for **{file.name}** must be a finite non-negative number "
                    "or left blank when the initial target is unknown."
                ),
            )
        target_values.append(to_excel_number(parsed_target))

    actual_values: list[int | float] = []
    for file, raw_actual in zip(sorted_files, actual_inputs, strict=True):
        parsed_actual = parse_required_number(raw_actual)
        if parsed_actual is None or parsed_actual < 0:
            return (
                {},
                [],
                [],
                [],
                [],
                f"{PREP_ACTUAL_CONCENTRATION_HEADER} for **{file.name}** must be a finite non-negative number.",
            )
        actual_values.append(to_excel_number(parsed_actual))

    source_txt_filenames = [file.name for file in sorted_files]
    treatment_inputs = special_treatment_inputs or [""] * len(sorted_files)
    special_treatments = ["" if value == "None" else value.strip() for value in treatment_inputs]

    return metadata, target_values, actual_values, source_txt_filenames, special_treatments, None


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
    required_keys = {"raman_shift", "intensity_columns"}
    if not required_keys.issubset(payload.keys()):
        st.session_state.pop(MERGED_PREVIEW_KEY, None)
        return None
    n_signals = len(payload["intensity_columns"])
    has_concentrations = (
        "target_concentrations" in payload
        and "actual_concentrations" in payload
        and len(payload["target_concentrations"]) == n_signals
        and len(payload["actual_concentrations"]) == n_signals
    )
    has_legacy_labels = "labels" in payload and len(payload["labels"]) == n_signals
    if not has_concentrations and not has_legacy_labels:
        st.session_state.pop(MERGED_PREVIEW_KEY, None)
        return None
    return payload


def _format_merged_preview_legend_label(
    target: int | float | str,
    actual: int | float,
    *,
    source_txt_filename: str = "",
) -> str:
    """Format concentrations and source TXT name for the merged-signal preview legend."""

    target_display = target if str(target).strip() != "" else "—"
    base = f"Target: {target_display} / Actual: {actual}"
    if source_txt_filename:
        return f"{base} ({source_txt_filename})"
    return base


def _normalize_legacy_preview_label(label: str) -> str:
    """Upgrade older preview legend text to the colon-separated format."""

    match = re.match(
        r"Target[:\s]+([^/]+?)\s*/\s*Actual[:\s]+(.+)$",
        label.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return label
    return _format_merged_preview_legend_label(match.group(1).strip(), match.group(2).strip())


def _merged_preview_legend_labels(preview_payload: dict[str, Any]) -> list[str]:
    """Return legend labels for each merged signal column."""

    if "target_concentrations" in preview_payload and "actual_concentrations" in preview_payload:
        source_names = preview_payload.get("source_txt_filenames", [])
        return [
            _format_merged_preview_legend_label(
                target,
                actual,
                source_txt_filename=(source_names[idx] if idx < len(source_names) else ""),
            )
            for idx, (target, actual) in enumerate(
                zip(
                    preview_payload["target_concentrations"],
                    preview_payload["actual_concentrations"],
                    strict=True,
                )
            )
        ]
    return [_normalize_legacy_preview_label(label) for label in preview_payload["labels"]]


def _merged_preview_to_tidy_dataframe(preview_payload: dict[str, Any]) -> pd.DataFrame:
    """Convert merged preview arrays to the tidy schema required by ``plot_spectra``."""

    raman_shift = preview_payload["raman_shift"]
    legend_labels = _merged_preview_legend_labels(preview_payload)
    frames = [
        pd.DataFrame(
            {
                "raman_shift": raman_shift,
                "intensity": intensities,
                "filename": f"signal_{signal_index}",
                "signal_index": signal_index,
                MERGED_PREVIEW_HUE_COL: label,
            }
        )
        for signal_index, (intensities, label) in enumerate(
            zip(preview_payload["intensity_columns"], legend_labels, strict=True)
        )
    ]
    return pd.concat(frames, ignore_index=True)


def _render_merged_preview(preview_payload: dict[str, Any]) -> None:
    """Plot merged TXT spectra with the shared ``plot_spectra`` styling."""

    st.markdown("---")
    st.markdown("### Preview of Merged Signals")
    tidy_df = _merged_preview_to_tidy_dataframe(preview_payload)
    fig = plot_spectra(
        tidy_df,
        hue=MERGED_PREVIEW_HUE_COL,
        figsize=MERGED_PREVIEW_FIGSIZE,
        title="Merged Signal Preview",
    )
    legend = fig.axes[0].get_legend()
    if legend is not None:
        legend.set_title(MERGED_PREVIEW_LEGEND_TITLE)
        for text in legend.get_texts():
            text.set_fontsize(8)
    render_figure_stretch(fig)


@st.fragment
def _render_prep_export_and_preview() -> None:
    """
    Render post-convert download and spectrum preview in an isolated fragment.

    Keeps download clicks and preview redraws from rerunning metadata widgets,
    which reduces horizontal layout jitter on some browsers.
    """

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
            on_click="ignore",
        )

    preview_payload = _get_merged_preview()
    if preview_payload is not None:
        _render_merged_preview(preview_payload)


def _process_uploaded_template(
    uploaded_template: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
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

    if payload.get("version") != TEMPLATE_VERSION or not isinstance(payload.get("fields"), dict):
        return {
            "level": "error",
            "message": "Template requires supported version 1 and a fields object.",
            "warnings": [],
        }, None
    before_values = _collect_metadata_values_by_widget_key()
    updated_values, warnings = apply_template_import_to_values(payload, before_values)
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
    if payload.get("version") == TEMPLATE_VERSION and "signal_labels" in payload:
        signal_labels, signal_warnings = validate_signal_preset(payload["signal_labels"])
        warnings.extend(signal_warnings)
        updated_values[PER_SIGNAL_PRESET_KEY] = signal_labels
        message += f" Loaded labels for {len(signal_labels)} source filename(s)."
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
                st.session_state[selection_key] = is_template_field_exportable(
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
        include_signals = st.checkbox("Include per-signal Sample Type and Treatment", value=True)
        st.caption(
            "Signal labels restore only for matching source filenames. Concentrations are not stored."
        )
        signal_labels = {}
        if include_signals:
            for key, value in st.session_state.items():
                prefix = "txt2excel_sample_type_"
                if key.startswith(prefix) and value in SAMPLE_TYPE_OPTIONS:
                    filename = key[len(prefix) :]
                    signal_labels[filename] = {
                        "sample_type": value,
                        "special_treatment": st.session_state.get(
                            f"txt2excel_treatment_{filename}", "None"
                        ),
                    }
        export_payload = build_template_export_payload(field_values, selected_keys, signal_labels)
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


def _render_metadata_field_widget(
    column: DeltaGenerator,
    field_number: int,
    excel_label: str,
    widget_key: str,
    *,
    multiline: bool = False,
) -> None:
    """Render a single metadata widget using the appropriate Streamlit input type."""

    label = _metadata_field_label(field_number, excel_label)
    if field_number in DATE_METADATA_FIELD_NUMBERS:
        if widget_key not in st.session_state:
            st.session_state[widget_key] = None
        column.date_input(label, key=widget_key, format="YYYY-MM-DD")
        return
    if field_number in TIME_METADATA_FIELD_NUMBERS:
        if widget_key not in st.session_state:
            st.session_state[widget_key] = None
        column.time_input(label, key=widget_key)
        return
    if multiline:
        column.text_area(label, key=widget_key, height=100)
        return
    column.text_input(label, key=widget_key)


def _render_metadata_fields_in_columns(
    columns: list[DeltaGenerator],
    spec_by_number: dict[int, tuple[int, str, str]],
    field_numbers: tuple[int, ...],
    *,
    col_offset: int = 0,
) -> None:
    """
    Place metadata widgets into an existing column row starting at ``col_offset``.

    Parameters
    ----------
    columns:
        Column containers from ``st.columns(METADATA_UI_COLUMN_COUNT)``.
    spec_by_number:
        Mapping of field number to metadata spec triples.
    field_numbers:
        Field numbers to render in order.
    col_offset:
        Zero-based column index where the first field is placed.
    """

    for field_idx, field_number in enumerate(field_numbers):
        col_idx = col_offset + field_idx
        if col_idx >= len(columns):
            break
        number, excel_label, widget_key = spec_by_number[field_number]
        _render_metadata_field_widget(
            columns[col_idx],
            number,
            excel_label,
            widget_key,
        )


def _inject_prep_layout_stability_css() -> None:
    """
    Inject one-time CSS to reduce horizontal layout shift during Streamlit reruns.

    Reserving scrollbar gutter space helps prevent five-column metadata rows from
    pulsing wider/narrower when page height changes (common on Windows Chrome).
    """

    if st.session_state.get(_PREP_LAYOUT_STABILITY_CSS_KEY):
        return
    st.markdown(
        """
        <style>
        div[data-testid="stAppViewContainer"] {
            overflow-y: scroll;
        }
        div[data-testid="column"] {
            min-width: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_PREP_LAYOUT_STABILITY_CSS_KEY] = True


def _render_metadata_fields(container: DeltaGenerator) -> None:
    """
    Render numbered metadata in three five-column rows plus full-width notes.

    Row layout follows ``METADATA_UI_ROWS`` (5 + 5 + 5); Excel uses the same
    field order with dividers at each ``METADATA_LOGICAL_GROUPS`` boundary.
    """

    _inject_prep_layout_stability_css()
    container.markdown("### Experiment metadata")
    if st.session_state.pop(RESTORE_METADATA_AFTER_RELOAD_KEY, False):
        _restore_persistent_metadata_widgets()
    _apply_pending_template_import_values()
    spec_by_number = {
        field_number: (field_number, excel_label, widget_key)
        for field_number, excel_label, widget_key in METADATA_FIELD_SPECS
    }

    for row_fields in METADATA_UI_ROWS:
        row_columns = container.columns(METADATA_UI_COLUMN_COUNT)
        _render_metadata_fields_in_columns(row_columns, spec_by_number, row_fields)

    notes_field_number = METADATA_LOGICAL_GROUPS[-1][0]
    number, excel_label, widget_key = spec_by_number[notes_field_number]
    _render_metadata_field_widget(
        container,
        number,
        excel_label,
        widget_key,
        multiline=True,
    )

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

    common_shift, shift_error = validate_common_shift(file_contents)
    if shift_error or not uploaded_files:
        return common_shift, shift_error

    default_max_shift = str(common_shift.max()) if common_shift is not None else ""
    file_signature = tuple(
        sorted(
            (name, hashlib.sha256(content.encode()).hexdigest())
            for name, content in file_contents.items()
        )
    )
    if st.session_state.get("txt2excel_file_signature") != file_signature:
        for prefix in (
            "txt2excel_sample_type_",
            "txt2excel_treatment_",
            "txt2excel_target_",
            "txt2excel_actual_",
        ):
            _clear_session_keys_by_prefix(prefix)
        st.session_state.pop(MERGED_PREVIEW_KEY, None)
        st.session_state.pop(EXPORT_BYTES_KEY, None)
        st.session_state.pop(EXPORT_FILENAME_KEY, None)
        st.session_state.pop(EXPORT_CONTEXT_KEY, None)
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
            _prep_field_label("Min Raman Shift"),
            key="txt2excel_min_shift",
        )
    with col_max:
        col_max.text_input(
            _prep_field_label("Max Raman Shift"),
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
        seen_names: set[str] = set()
        for file in uploaded_files:
            identity = unicodedata.normalize("NFC", file.name).casefold()
            if identity in seen_names:
                st.error(f"Duplicate TXT filename: {file.name}. Use distinct filenames.")
                return
            seen_names.add(identity)
            try:
                file_contents[file.name] = file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                st.error(f"TXT file must use UTF-8 encoding: {file.name}")
                return

    shift_error: str | None = None
    if uploaded_files:
        _, shift_error = _sync_shift_defaults_for_upload(uploaded_files, file_contents)
        if shift_error:
            with st.sidebar:
                st.error(f"❌ {shift_error}")

    st.title("Raman TXT to Excel Merger")

    if not uploaded_files:
        st.info(
            "Upload one or more `.txt` files using the sidebar, then complete metadata "
            "required metadata (``*``), optional fields ``(optional)``, and per-signal rows below."
        )
        st.markdown("### Next steps")
        st.markdown(
            "1. Upload `.txt` files in the sidebar.\n"
            "2. Complete required metadata fields (marked with ``*``).\n"
            "3. Set the Raman shift range and per-signal concentrations (optional where noted).\n"
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

    _render_raman_shift_bounds(st)

    st.markdown("### Per-signal labels and concentrations")

    header_file, header_sample, header_treatment, header_target, header_actual = st.columns(
        [2, 1.3, 1.3, 1, 1]
    )
    with header_sample:
        st.markdown("**Sample Type***")
    with header_file:
        st.markdown("**File**")
    with header_treatment:
        st.markdown(_prep_column_header_markup(PREP_SPECIAL_TREATMENT_HEADER, optional=True))
    with header_target:
        st.markdown(_prep_column_header_markup(PREP_TARGET_CONCENTRATION_HEADER, optional=True))
    with header_actual:
        st.markdown(_prep_column_header_markup(PREP_ACTUAL_CONCENTRATION_HEADER))

    target_inputs: list[str] = []
    actual_inputs: list[str] = []
    special_treatment_inputs: list[str] = []
    sample_type_inputs: list[str] = []
    for file in sorted_files:
        col_file, col_sample, col_treatment, col_target, col_actual = st.columns(
            [2, 1.3, 1.3, 1, 1]
        )
        preset = st.session_state.get(PER_SIGNAL_PRESET_KEY, {}).get(file.name, {})
        sample_key = f"txt2excel_sample_type_{file.name}"
        treatment_key = f"txt2excel_treatment_{file.name}"
        if sample_key not in st.session_state:
            st.session_state[sample_key] = preset.get("sample_type") or None
        if treatment_key not in st.session_state:
            st.session_state[treatment_key] = preset.get("special_treatment", "None") or "None"
        with col_sample:
            sample_type_inputs.append(
                st.selectbox(
                    f"Sample Type for {file.name}",
                    SAMPLE_TYPE_OPTIONS,
                    index=None,
                    key=sample_key,
                    label_visibility="collapsed",
                    placeholder="Select sample type",
                )
            )
        with col_file:
            st.markdown(
                f"<div style='padding-top:6px'>{escape(file.name)}</div>",
                unsafe_allow_html=True,
            )
        with col_treatment:
            special_treatment_inputs.append(
                st.selectbox(
                    label=f"Treatment for {file.name}",
                    options=SPECIAL_TREATMENT_OPTIONS,
                    key=treatment_key,
                    label_visibility="collapsed",
                )
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
    st.markdown("### Output File Name")
    col_name_input, col_ext = st.columns([9, 1])
    with col_name_input:
        output_file_name = st.text_input(
            "Output File Name",
            placeholder="Enter output file name",
            key="txt2excel_output_name",
            label_visibility="collapsed",
        )
    with col_ext:
        st.markdown(
            "<div style='padding-top:6px;font-weight:600'>.xlsx</div>",
            unsafe_allow_html=True,
        )

    convert_clicked = st.button(
        "Convert and Export",
        key="txt2excel_convert_button",
        type="primary",
    )

    current_context = (
        tuple(
            (name, hashlib.sha256(content.encode()).hexdigest())
            for name, content in file_contents.items()
        ),
        tuple(_collect_metadata_values().items()),
        tuple(target_inputs),
        tuple(actual_inputs),
        tuple(special_treatment_inputs),
        tuple(sample_type_inputs),
        _read_shift_bounds(),
        output_file_name.strip(),
    )
    if st.session_state.get(EXPORT_CONTEXT_KEY) != current_context:
        for key in (MERGED_PREVIEW_KEY, EXPORT_BYTES_KEY, EXPORT_FILENAME_KEY):
            st.session_state.pop(key, None)

    if convert_clicked:
        if any(value not in SAMPLE_TYPE_OPTIONS for value in sample_type_inputs):
            st.error("Select Sample Type for every signal before exporting.")
            return
        if not output_file_name.strip():
            st.warning("Please enter an output file name.")
            return

        (
            metadata,
            target_values,
            actual_values,
            source_txt_filenames,
            special_treatments,
            validation_error,
        ) = _validate_prep_inputs(
            sorted_files,
            target_inputs,
            actual_inputs,
            special_treatment_inputs=special_treatment_inputs,
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
        raman_shift, intensity_columns, merge_error = merge_txt_spectra(
            ordered_contents,
            min_shift=min_shift,
            max_shift=max_shift,
        )
        if merge_error or raman_shift is None or intensity_columns is None:
            st.error(f"❌ {merge_error or 'Merge failed.'}")
            st.session_state.pop(MERGED_PREVIEW_KEY, None)
            st.session_state.pop(EXPORT_BYTES_KEY, None)
            st.session_state.pop(EXPORT_FILENAME_KEY, None)
            st.session_state.pop(EXPORT_CONTEXT_KEY, None)
            return

        workbook_rows = build_embedded_workbook_rows(
            metadata,
            target_concentrations=target_values,
            actual_concentrations=actual_values,
            source_txt_filenames=source_txt_filenames,
            special_treatments=special_treatments,
            sample_types=sample_type_inputs,
            raman_shift=raman_shift,
            intensity_columns=intensity_columns,
        )
        excel_bytes = embedded_workbook_to_excel_bytes(workbook_rows)
        export_filename = f"{output_file_name.strip()}.xlsx"

        st.session_state[MERGED_PREVIEW_KEY] = {
            "raman_shift": raman_shift,
            "intensity_columns": intensity_columns,
            "target_concentrations": target_values,
            "actual_concentrations": actual_values,
            "source_txt_filenames": source_txt_filenames,
            "special_treatments": special_treatments,
            "sample_types": sample_type_inputs,
        }
        st.session_state[EXPORT_BYTES_KEY] = excel_bytes
        st.session_state[EXPORT_FILENAME_KEY] = export_filename
        st.session_state[EXPORT_CONTEXT_KEY] = current_context
        logger.info(
            "Embedded TXT merge complete: %d files → %d Raman rows, %d signal columns",
            len(sorted_files),
            len(raman_shift),
            len(intensity_columns),
        )

    _render_prep_export_and_preview()
