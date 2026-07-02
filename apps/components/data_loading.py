"""
Data loading utilities for SERS Data Explorer.
"""

import logging
import uuid
from zipfile import BadZipFile

import pandas as pd
import streamlit as st

from sensd_sers_analysis.application.contracts import LoadedDataBundle
from sensd_sers_analysis.application.dataset_pipeline import load_uploaded_bundle
from sensd_sers_analysis.data.io import SersLoadReport

from state import reset_ui_state

logger = logging.getLogger(__name__)

# Session-state key controlling file_uploader identity. Incrementing it forces
# the uploader to remount as a new widget, clearing its files.
UPLOADER_RESET_KEY = "_uploader_reset"


def clear_app_data() -> None:
    """
    Reset the app's data state for a clean upload.

    Clears st.cache_data memoized dataframes and all st.session_state keys
    so filter states, file uploader state, and UI flags are wiped. Sets a new
    uploader reset key so the file_uploader remounts with no files.
    """
    logger.info("Clearing app data (Reload Data clicked)")
    st.cache_data.clear()
    reset_ui_state()
    st.session_state[UPLOADER_RESET_KEY] = str(uuid.uuid4())


def _upload_failure_report(
    files_data: tuple[tuple[str, bytes], ...],
    message: str,
) -> SersLoadReport:
    """Build a load report when the entire upload batch fails before parsing."""

    return SersLoadReport(
        loaded_files=(),
        skipped_files=tuple((name, message) for name, _ in files_data),
    )


def _render_skipped_files_expander(
    container: st.delta_generator.DeltaGenerator,
    report: SersLoadReport,
) -> None:
    """List skipped workbooks inside a collapsible expander."""

    with container.expander(f"Could not load {report.n_skipped} file(s)") as expander:
        for filename, message in report.skipped_files:
            expander.markdown(f"**{filename}**")
            expander.caption(message)


def render_upload_load_status(
    loaded_bundle: LoadedDataBundle,
    *,
    uploaded_count: int,
    sidebar: st.delta_generator.DeltaGenerator | None = None,
) -> None:
    """
    Show a single user-facing upload status block in the sidebar.

    Parameters
    ----------
    loaded_bundle:
        Parsed upload bundle including per-file load outcomes.
    uploaded_count:
        Number of files selected in the uploader widget.
    sidebar:
        Streamlit container for the status block.
    """

    sidebar_panel = sidebar if sidebar is not None else st.sidebar
    report = loaded_bundle.load_report
    n_samples = len(loaded_bundle.wide_df)
    n_tidy = len(loaded_bundle.tidy_df)
    has_data = not loaded_bundle.tidy_df.empty

    if report.n_skipped == 0 and has_data:
        sidebar_panel.success(
            "Loaded "
            f"**{report.n_loaded or uploaded_count}** of **{uploaded_count}** files, "
            f"**{n_samples}** samples ({n_tidy} tidy rows)."
        )
        return

    if report.n_loaded > 0 and has_data:
        sidebar_panel.warning(
            "Loaded "
            f"**{report.n_loaded}** of **{uploaded_count}** files, "
            f"**{n_samples}** samples ({n_tidy} tidy rows). "
            f"**{report.n_skipped}** file(s) could not be read."
        )
        _render_skipped_files_expander(sidebar_panel, report)
        return

    sidebar_panel.error(f"Could not load any of the **{uploaded_count}** uploaded files.")
    _render_skipped_files_expander(sidebar_panel, report)


@st.cache_data
def load_from_uploaded(
    _files_data: tuple[tuple[str, bytes], ...],
) -> LoadedDataBundle:
    """
    Load SERS data from uploaded file bytes and convert to tidy format.

    Args:
        _files_data: Tuple of (filename, file_bytes) per uploaded file.
            Leading underscore to exclude from Streamlit's cache key display.

    Returns:
        Parsed upload bundle. Empty dataframes are returned on failure.
    """
    if not _files_data:
        logger.warning("load_from_uploaded called with no files")
        return LoadedDataBundle(wide_df=pd.DataFrame(), tidy_df=pd.DataFrame())
    logger.info("Loading %d uploaded file(s): %s", len(_files_data), [n for n, _ in _files_data])
    try:
        bundle = load_uploaded_bundle(_files_data)
    except BadZipFile:
        message = (
            "This file is not a valid Excel workbook (.xlsx). "
            "It may be corrupt or saved in an unsupported format."
        )
        logger.warning("Upload parsing failed: invalid zip/xlsx container")
        return LoadedDataBundle(
            wide_df=pd.DataFrame(),
            tidy_df=pd.DataFrame(),
            load_report=_upload_failure_report(_files_data, message),
        )
    except OSError as exc:
        message = f"The upload could not be read from disk. Details: {exc}"
        logger.warning("Upload parsing failed: %s", exc)
        return LoadedDataBundle(
            wide_df=pd.DataFrame(),
            tidy_df=pd.DataFrame(),
            load_report=_upload_failure_report(_files_data, message),
        )
    except Exception as exc:
        message = f"An unexpected error occurred while reading the uploaded files. Details: {exc}"
        logger.exception("Upload parsing failed")
        return LoadedDataBundle(
            wide_df=pd.DataFrame(),
            tidy_df=pd.DataFrame(),
            load_report=_upload_failure_report(_files_data, message),
        )

    if bundle.wide_df.empty or bundle.tidy_df.empty:
        if bundle.load_report.n_skipped == 0:
            logger.warning("load_uploaded_bundle returned empty dataframes")
    return bundle
