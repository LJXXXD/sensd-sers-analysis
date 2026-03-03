"""
Raman shift and peaks-per-serotype sidebar component.
"""

import streamlit as st

from sensd_sers_analysis.utils import parse_raman_shift_bound

from theme import N_PEAKS_DEFAULT, N_PEAKS_MAX, N_PEAKS_MIN


def render_raman_and_peaks_sidebar(
    container, wide_df
) -> tuple[float | None, float | None, int, dict | None]:
    """
    Render Raman shift inputs and peaks-per-serotype controls in the sidebar.

    Args:
        container: Streamlit container (typically st.sidebar).
        wide_df: Wide-format DataFrame (may be None or empty).

    Returns:
        Tuple of (min_shift, max_shift, n_peaks, n_peaks_by_serotype).
        n_peaks_by_serotype is None when no serotypes; otherwise dict of
        serotype -> int.
    """
    container.markdown("#### Raman Shift window")
    rs_col1, rs_col2 = container.columns(2)
    with rs_col1:
        rs_min_str = st.text_input(
            "Min (cm⁻¹)",
            value="",
            placeholder="e.g. 400",
            key="raman_shift_min",
            help="Leave blank for no lower limit.",
        )
    with rs_col2:
        rs_max_str = st.text_input(
            "Max (cm⁻¹)",
            value="",
            placeholder="e.g. 1800",
            key="raman_shift_max",
            help="Leave blank for no upper limit.",
        )

    serotypes_from_wide = (
        sorted(wide_df["serotype"].dropna().unique().astype(str).tolist())
        if wide_df is not None and not wide_df.empty and "serotype" in wide_df.columns
        else []
    )
    serotypes_from_wide = [s for s in serotypes_from_wide if s and s != "nan"]

    if serotypes_from_wide:
        n_peaks = N_PEAKS_DEFAULT
        container.markdown("#### Peaks per serotype")
        n_peaks_by_serotype = {}
        for i, s in enumerate(serotypes_from_wide):
            if i % 2 == 0:
                peak_cols = container.columns(2)
            with peak_cols[i % 2]:
                n_peaks_by_serotype[s] = int(
                    st.number_input(
                        f"Peaks ({s})",
                        min_value=N_PEAKS_MIN,
                        max_value=N_PEAKS_MAX,
                        value=N_PEAKS_DEFAULT,
                        step=1,
                        key=f"n_peaks_{s}",
                        help=f"Peaks for {s}",
                    )
                )
    else:
        n_peaks_by_serotype = None
        n_peaks = int(
            container.number_input(
                "Number of Peaks",
                min_value=N_PEAKS_MIN,
                max_value=N_PEAKS_MAX,
                value=N_PEAKS_DEFAULT,
                step=1,
                key="n_peaks",
                help="Number of peaks (no serotype column in data).",
            )
        )

    min_shift = parse_raman_shift_bound(rs_min_str)
    max_shift = parse_raman_shift_bound(rs_max_str)
    return min_shift, max_shift, n_peaks, n_peaks_by_serotype
