"""
Raman shift sidebar component (trim window and global peak count when needed).
"""

from __future__ import annotations

import streamlit as st

from sensd_sers_analysis.processing.metadata import sorted_unique_canonical_serotypes
from sensd_sers_analysis.utils import parse_raman_shift_bound

from theme import (
    N_PEAKS_DEFAULT,
    N_PEAKS_MAX,
    N_PEAKS_MIN,
    RAMAN_SHIFT_DEFAULT_MAX_CM1,
    RAMAN_SHIFT_DEFAULT_MIN_CM1,
)


def list_serotypes_from_wide_df(wide_df) -> list[str]:
    """
    Return sorted unique serotype labels from a wide dataframe.

    Matching is case-insensitive; returned labels are canonical uppercase.

    Parameters
    ----------
    wide_df:
        Wide-format dataframe; may be empty.

    Returns
    -------
    list[str]
        Non-empty canonical serotype strings, sorted.
    """

    return sorted_unique_canonical_serotypes(wide_df)


def render_raman_shift_sidebar(container, wide_df) -> tuple[float | None, float | None, int]:
    """
    Render Raman shift trim inputs and, when there is no serotype column, peak count.

    Per-serotype peak counts for **Peak Discovery** are configured in that
    tab (above each serotype plot). When the data has no ``serotype`` column, a
    single **Number of peaks** control is shown here.

    Parameters
    ----------
    container:
        Streamlit container (typically ``st.sidebar``).
    wide_df:
        Loaded wide dataframe (used only to decide if the global peak control is shown).

    Returns
    -------
    tuple[float | None, float | None, int]
        ``(min_shift, max_shift, n_peaks)``. When serotypes are present, ``n_peaks``
        is the package default (dynamic counts come from session state in the app).
        When there are no serotypes, ``n_peaks`` is taken from the sidebar control.
    """

    container.markdown("#### Raman shift window")
    rs_col1, rs_col2 = container.columns(2)
    with rs_col1:
        rs_min_str = st.text_input(
            "Min (cm⁻¹)",
            value=str(RAMAN_SHIFT_DEFAULT_MIN_CM1),
            key="raman_shift_min",
            help="Lower bound for trimming spectra. Clear the field for no lower limit.",
        )
    with rs_col2:
        rs_max_str = st.text_input(
            "Max (cm⁻¹)",
            value=str(RAMAN_SHIFT_DEFAULT_MAX_CM1),
            key="raman_shift_max",
            help="Upper bound for trimming spectra. Clear the field for no upper limit.",
        )

    serotypes = list_serotypes_from_wide_df(wide_df)
    if serotypes:
        n_peaks = int(N_PEAKS_DEFAULT)
    else:
        n_peaks = int(
            container.number_input(
                "Number of peaks",
                min_value=N_PEAKS_MIN,
                max_value=N_PEAKS_MAX,
                value=N_PEAKS_DEFAULT,
                step=1,
                key="n_peaks_global_no_serotype",
                help="Number of peaks when the dataset has no serotype column.",
            )
        )

    min_shift = parse_raman_shift_bound(rs_min_str)
    max_shift = parse_raman_shift_bound(rs_max_str)
    return min_shift, max_shift, n_peaks
