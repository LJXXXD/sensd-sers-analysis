"""
Data loading utilities for SERS Data Explorer.
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from sensd_sers_analysis.data import load_sers_data_as_wide_and_tidy


@st.cache_data
def load_from_uploaded(
    _files_data: tuple[tuple[str, bytes], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load SERS data from uploaded file bytes and convert to tidy format.

    Args:
        _files_data: Tuple of (filename, file_bytes) per uploaded file.
            Leading underscore to exclude from Streamlit's cache key display.

    Returns:
        Tuple of (wide_df, tidy_df). Empty DataFrames if loading fails.
    """
    if not _files_data:
        return pd.DataFrame(), pd.DataFrame()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        paths = [str(tmp_path / name) for name, _ in _files_data]
        for (name, content), p in zip(_files_data, paths):
            Path(p).write_bytes(content)
        return load_sers_data_as_wide_and_tidy(paths)
