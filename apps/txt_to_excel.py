"""
Instrument TXT to Excel merger — Streamlit prep utility.

Converts raw Raman instrument ``.txt`` exports into embedded-metadata Excel
workbooks for the SERS analysis tool. Lives entirely under ``apps/``.
"""

from __future__ import annotations

import logging
import re
import uuid
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
DEFAULT_MIN_SHIFT = 560.9

TARGET_CONCENTRATION_LABEL = "Target Concentration (CFU/mL)"
ACTUAL_CONCENTRATION_LABEL = "Actual Concentration (CFU/mL)"
INTENSITY_HEADER = "Relative Light intensity (a.u)"
RAMAN_SHIFT_HEADER = "Raman Shift"

# Metadata field numbers whose column-B values must be numeric in the workbook.
NUMERIC_METADATA_FIELD_NUMBERS = frozenset({2, 3, 4, 5, 6, 7})
METADATA_FIELD_SPECS: tuple[tuple[int, str, str], ...] = (
    (1, "Testing Plan", "txt2excel_meta_testing_plan"),
    (2, "Disk Diameter (nm)", "txt2excel_meta_disk_diameter_nm"),
    (3, "Periodicity (µm)", "txt2excel_meta_periodicity_um"),
    (4, "Thickness (nm)", "txt2excel_meta_thickness_nm"),
    (5, "Core Diameter (µm)", "txt2excel_meta_core_diameter_um"),
    (6, "Integration Time (ms):", "txt2excel_meta_integration_time_ms"),
    (7, "Scan Average:", "txt2excel_meta_scan_average"),
    (8, "Sensor Model", "txt2excel_meta_sensor_model"),
    (9, "Sensor ID", "txt2excel_meta_sensor_id"),
    (10, "Test ID", "txt2excel_meta_test_id"),
    (11, "Connection ID", "txt2excel_meta_connection_id"),
    (12, "Serotype", "txt2excel_meta_serotype"),
    (13, "Date", "txt2excel_meta_date"),
    (14, "Testing Time", "txt2excel_meta_testing_time"),
    (15, "Operator", "txt2excel_meta_operator"),
    (16, "Rinsate Type", "txt2excel_meta_rinsate_type"),
)


def enter_prep_mode() -> None:
    """Switch the app shell to the TXT-to-Excel prep utility."""

    st.session_state[APP_MODE_KEY] = APP_MODE_PREP


def enter_analysis_mode() -> None:
    """Return to the main SERS analysis explorer."""

    st.session_state[APP_MODE_KEY] = APP_MODE_ANALYSIS
    st.session_state.pop(MERGED_PREVIEW_KEY, None)
    st.session_state.pop(EXPORT_BYTES_KEY, None)
    st.session_state.pop(EXPORT_FILENAME_KEY, None)


def clear_prep_uploads() -> None:
    """
    Reset uploaded TXT files and derived export artifacts.

    Metadata field inputs are preserved so collaborators can reuse experiment details.
    """

    logger.info("Clearing prep upload state (Reload Data clicked)")
    st.session_state.pop(MERGED_PREVIEW_KEY, None)
    st.session_state.pop(EXPORT_BYTES_KEY, None)
    st.session_state.pop(EXPORT_FILENAME_KEY, None)
    st.session_state.pop("txt2excel_file_signature", None)
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


def _collect_metadata_values() -> dict[str, str]:
    """Read metadata field values from Streamlit session state."""

    return {
        excel_label: str(st.session_state.get(widget_key, "")).strip()
        for _, excel_label, widget_key in METADATA_FIELD_SPECS
    }


def _validate_metadata(metadata: dict[str, str]) -> str | None:
    """Return an error message when any required metadata field is empty."""

    missing = [label for label, value in metadata.items() if not value]
    if missing:
        return f"All metadata fields are required. Missing: {', '.join(missing)}"
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

    metadata_raw = _collect_metadata_values()
    metadata_error = _validate_metadata(metadata_raw)
    if metadata_error:
        return {}, [], [], metadata_error

    metadata: dict[str, Any] = {}
    for field_number, excel_label, _ in METADATA_FIELD_SPECS:
        raw_value = metadata_raw[excel_label]
        if field_number in NUMERIC_METADATA_FIELD_NUMBERS:
            parsed = _parse_required_number(raw_value)
            if parsed is None:
                return (
                    {},
                    [],
                    [],
                    f"{field_number}. {excel_label} must be a number.",
                )
            metadata[excel_label] = _to_excel_number(parsed)
        else:
            metadata[excel_label] = raw_value

    target_values: list[int | float] = []
    for file, raw_target in zip(sorted_files, target_inputs, strict=True):
        parsed_target = _parse_required_number(raw_target)
        if parsed_target is None:
            return (
                {},
                [],
                [],
                (
                    f"17. Target (CFU/mL) for **{file.name}** must be a number. "
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
                f"18. Actual (CFU/mL) for **{file.name}** must be a number.",
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
_SEPARATOR_AFTER_METADATA_NUMBERS = frozenset({5, 7, 15})


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


def _render_metadata_fields(container: DeltaGenerator) -> None:
    """Render numbered metadata inputs in four columns, filled row by row."""

    container.markdown("### Experiment metadata")
    container.caption(
        "Fields 1–16 are required and persist after export. Fields 2–7 must be numbers."
    )
    n_columns = 4
    for row_start in range(0, len(METADATA_FIELD_SPECS), n_columns):
        row_specs = METADATA_FIELD_SPECS[row_start : row_start + n_columns]
        columns = container.columns(n_columns)
        for column, (number, excel_label, widget_key) in zip(columns, row_specs, strict=False):
            column.text_input(
                f"{number}. {excel_label}",
                key=widget_key,
            )


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
    """Render min/max Raman shift inputs above experiment metadata."""

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
            "fields 1–16 and concentrations 17–18 below."
        )
        st.markdown("### Next steps")
        st.markdown(
            "1. Upload `.txt` files in the sidebar.\n"
            "2. Set the Raman shift range and complete metadata fields 1–16.\n"
            "3. Enter target and actual concentrations for each file.\n"
            "4. Convert and download the embedded `.xlsx`.\n"
            "5. Switch back to **Analysis** and upload the saved file."
        )
        return

    if shift_error:
        st.warning("Fix Raman shift mismatches before continuing.")
        return

    _render_raman_shift_bounds(st)
    st.markdown("---")
    _render_metadata_fields(st)
    st.markdown("---")

    sorted_files = sorted(uploaded_files, key=lambda file: _extract_cfu_sort_key(file.name))
    n_files = len(sorted_files)

    st.markdown("### Concentrations per spectrum")
    st.caption(
        f"{n_files} signal column{'s' if n_files != 1 else ''} will be written. "
        "Target and actual concentrations must be numbers. Use **0** for rinsate-only controls."
    )

    header_file, header_target, header_actual = st.columns([2, 1, 1])
    with header_file:
        st.markdown("**File**")
    with header_target:
        st.markdown("**17. Target (CFU/mL)**")
    with header_actual:
        st.markdown("**18. Actual (CFU/mL)**")

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
