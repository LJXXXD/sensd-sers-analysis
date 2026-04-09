"""
Typed Streamlit state adapters for the SERS Data Explorer.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import streamlit as st

from sensd_sers_analysis.application.contracts import PeakArtifacts

PEAK_ARTIFACTS_SESSION_KEY = "peak_artifacts"


def get_filter_widget_key(column: str) -> str:
    """
    Build the widget key for a canonical filter column.

    Parameters
    ----------
    column:
        Canonical dataframe column name.

    Returns
    -------
    str
        Stable widget key for the filter selector.
    """

    return f"filter_{column}"


def get_filter_exclude_widget_key(column: str) -> str:
    """
    Build the widget key for a filter's exclude toggle.

    Parameters
    ----------
    column:
        Canonical dataframe column name.

    Returns
    -------
    str
        Stable widget key for the exclude toggle.
    """

    return f"{get_filter_widget_key(column)}_exclude"


def clear_filter_widget_state(column: str) -> None:
    """
    Clear one filter widget and its exclude toggle.

    Parameters
    ----------
    column:
        Canonical dataframe column name.
    """

    st.session_state[get_filter_widget_key(column)] = []
    st.session_state[get_filter_exclude_widget_key(column)] = False


def clear_all_filter_widget_state(columns: Iterable[str]) -> None:
    """
    Clear all filter widgets for the supplied columns.

    Parameters
    ----------
    columns:
        Canonical dataframe column names.
    """

    for column in columns:
        clear_filter_widget_state(column)


def write_peak_artifacts_to_state(peak_artifacts: PeakArtifacts) -> None:
    """
    Store peak artifacts in Streamlit session state.

    Parameters
    ----------
    peak_artifacts:
        Peak artifacts to persist for the current session.
    """

    st.session_state[PEAK_ARTIFACTS_SESSION_KEY] = peak_artifacts


def read_peak_artifacts_from_state() -> PeakArtifacts:
    """
    Read peak artifacts from Streamlit session state.

    Returns
    -------
    PeakArtifacts
        Stored peak artifacts, or an empty placeholder when absent.
    """

    if PEAK_ARTIFACTS_SESSION_KEY in st.session_state:
        return st.session_state[PEAK_ARTIFACTS_SESSION_KEY]
    return PeakArtifacts(
        peak_infos_by_serotype={},
        mean_spec_by_serotype={},
        default_serotype=None,
        raman_x=np.array([], dtype=float),
    )


def clear_peak_artifacts_from_state() -> None:
    """Remove peak artifacts from session state if present."""

    st.session_state.pop(PEAK_ARTIFACTS_SESSION_KEY, None)


def reset_ui_state(*, preserve_keys: Iterable[str] = ()) -> None:
    """
    Reset Streamlit session state while optionally preserving selected keys.

    Parameters
    ----------
    preserve_keys:
        Keys to preserve across the reset.
    """

    preserved = {key: st.session_state[key] for key in preserve_keys if key in st.session_state}
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    for key, value in preserved.items():
        st.session_state[key] = value
