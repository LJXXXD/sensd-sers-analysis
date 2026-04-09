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
    except (BadZipFile, OSError, ValueError) as exc:
        logger.warning("Upload parsing failed: %s", exc)
        return LoadedDataBundle(
            wide_df=pd.DataFrame(),
            tidy_df=pd.DataFrame(),
        )

    if bundle.wide_df.empty or bundle.tidy_df.empty:
        logger.warning("load_uploaded_bundle returned empty dataframes")
    return bundle
