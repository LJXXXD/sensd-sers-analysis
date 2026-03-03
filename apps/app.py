"""
Self-Serve SERS Data Explorer — Streamlit UI for loading, filtering, and plotting SERS spectra.

Collaborators can load datasets, filter dynamically, and generate plots without writing code.
"""

import numpy as np
import streamlit as st

from sensd_sers_analysis.data import count_unique_spectra, wide_to_tidy
from sensd_sers_analysis.processing import (
    extract_basic_features,
    extract_dynamic_peak_features,
    filter_sers_data,
    get_filter_options,
    get_filterable_columns,
    get_peak_height_columns,
    preprocess_metadata,
    trim_raman_shift,
)
from sensd_sers_analysis.utils import format_column_label

from components.data_loading import load_from_uploaded
from components.raman_sidebar import render_raman_and_peaks_sidebar
from components.filter_ui import (
    MAIN_FILTER_COUNT,
    _FILTER_DIVIDER,
    _TITLE_TO_FILTER_DIVIDER,
    _render_filter,
    render_main_filter_header,
    section_divider,
)

from tabs import (
    feature_analysis,
    model_consistency,
    peak_diagnostics,
    sensor_assessment,
    serotype_classification,
    spectra_viewer,
)

st.set_page_config(
    page_title="SERS Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
st.sidebar.markdown("# 📁 Data Loading")
uploaded = st.sidebar.file_uploader(
    "Upload Excel (.xlsx) files",
    type=["xlsx", "xls"],
    accept_multiple_files=True,
)
wide_df = None
tidy_df = None
if uploaded:
    files_data = tuple((f.name, f.getvalue()) for f in uploaded)
    wide_df, tidy_df = load_from_uploaded(files_data)

if tidy_df is None or tidy_df.empty:
    st.info("👆 Load data using the sidebar: upload Excel (.xlsx) files.")
    st.stop()

tidy_df = preprocess_metadata(tidy_df)
wide_df = preprocess_metadata(wide_df)

st.sidebar.success(
    f"Loaded **{len(uploaded)}** files, **{len(wide_df)}** samples ({len(tidy_df)} tidy rows)."
)

# ---------------------------------------------------------------------------
# Raman shift trimming and peaks per serotype
# ---------------------------------------------------------------------------
min_shift, max_shift, n_peaks, n_peaks_by_serotype = render_raman_and_peaks_sidebar(
    st.sidebar, wide_df
)
wide_df = trim_raman_shift(wide_df, min_shift=min_shift, max_shift=max_shift)
tidy_df = wide_to_tidy(wide_df)
tidy_df = preprocess_metadata(tidy_df)

features_df = extract_basic_features(wide_df)

peak_df, peak_by_sero, mean_by_sero, default_sero, raman_x = (
    extract_dynamic_peak_features(
        wide_df, n_peaks=int(n_peaks), n_peaks_by_serotype=n_peaks_by_serotype
    )
)
if peak_by_sero:
    first_infos = next(iter(peak_by_sero.values()))
    peak_cols = get_peak_height_columns(first_infos)
    features_df = features_df.join(peak_df[peak_cols], how="left")
    st.session_state["peak_infos_by_serotype"] = peak_by_sero
    st.session_state["mean_spec_by_serotype"] = mean_by_sero
    st.session_state["peak_default_serotype"] = default_sero
    st.session_state["raman_x"] = raman_x
else:
    st.session_state["peak_infos_by_serotype"] = {}
    st.session_state["mean_spec_by_serotype"] = {}
    st.session_state["peak_default_serotype"] = None
    st.session_state["raman_x"] = np.array([])

st.sidebar.markdown(section_divider(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Render Filter UI -> Apply Filters (dynamic from metadata columns)
# ---------------------------------------------------------------------------
filter_columns = get_filterable_columns(tidy_df)

render_main_filter_header(st.sidebar, filter_columns)
st.sidebar.markdown(_TITLE_TO_FILTER_DIVIDER, unsafe_allow_html=True)

main_cols = filter_columns[:MAIN_FILTER_COUNT]
more_cols = filter_columns[MAIN_FILTER_COUNT:]

filter_state: dict[str, tuple[list | None, bool]] = {}

for i, col in enumerate(main_cols):
    if i > 0:
        st.sidebar.markdown(_FILTER_DIVIDER, unsafe_allow_html=True)
    opts_all = get_filter_options(tidy_df, filter_columns, filter_state)
    help_text = "Binned concentration." if col == "concentration_group" else ""
    selected, exclude = _render_filter(
        format_column_label(col),
        opts_all[col],
        [],
        False,
        st.sidebar,
        help_text=help_text,
        reset_button_key=f"reset_{col}",
    )
    filter_state[col] = (selected if selected else None, exclude)

with st.sidebar.expander("More Filters", expanded=False):
    for i, col in enumerate(more_cols):
        if i > 0:
            st.markdown(_FILTER_DIVIDER, unsafe_allow_html=True)
        opts_all = get_filter_options(tidy_df, filter_columns, filter_state)
        help_text = "Leave empty for no filter." if col == "filename" else ""
        selected, exclude = _render_filter(
            format_column_label(col),
            opts_all[col],
            [],
            False,
            st,
            help_text=help_text,
            reset_button_key=f"reset_more_{col}",
        )
        filter_state[col] = (selected if selected else None, exclude)

# Build filter_state for columns we didn't render (e.g. signal_index in filter but not in UI)
filter_state_for_apply = {k: v for k, v in filter_state.items() if k in filter_columns}
filtered = filter_sers_data(tidy_df, filter_state_for_apply)
filtered_features = filter_sers_data(features_df, filter_state_for_apply)

# ---------------------------------------------------------------------------
# 3. Main: Summary and Tabs
# ---------------------------------------------------------------------------
n_filtered = count_unique_spectra(filtered)
st.caption(
    f"Filtered to **{n_filtered}** spectrum traces, **{len(filtered_features)}** "
    "samples for feature analysis"
)

if filtered.empty:
    st.warning("No data matches the selected filters. Adjust filters and try again.")
    st.stop()

(
    tab_spectra,
    tab_peak_diag,
    tab_stats,
    tab_assessment,
    tab_model_consistency,
    tab_phase2,
) = st.tabs(
    [
        "Spectra Viewer",
        "Peak Diagnostics",
        "Feature Analysis",
        "Sensor Assessment",
        "Model Consistency",
        "Serotype Classification",
    ]
)

with tab_spectra:
    spectra_viewer.render(filtered)

with tab_peak_diag:
    peak_diagnostics.render(filtered_features, wide_df)

with tab_stats:
    feature_analysis.render(filtered_features)

with tab_assessment:
    sensor_assessment.render(filtered_features)

with tab_model_consistency:
    model_consistency.render(filtered_features)

with tab_phase2:
    serotype_classification.render(filtered_features)
