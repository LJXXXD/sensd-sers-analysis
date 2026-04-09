"""
Self-Serve SERS Data Explorer — Streamlit UI for loading, filtering, and plotting SERS spectra.

Collaborators can load datasets, filter dynamically, and generate plots without writing code.
"""

import logging

import streamlit as st

from cache import apply_cached_filters, build_cached_derived_bundle
from components.data_loading import (
    UPLOADER_RESET_KEY,
    clear_app_data,
    load_from_uploaded,
)
from components.filter_ui import (
    MAIN_FILTER_COUNT,
    _FILTER_DIVIDER,
    _TITLE_TO_FILTER_DIVIDER,
    _render_filter,
    render_main_filter_header,
    section_divider,
)
from components.raman_sidebar import render_raman_and_peaks_sidebar
from sensd_sers_analysis.application import (
    FilterSelection,
    build_filter_catalog,
    compute_filter_options,
    serialize_filter_state,
)
from sensd_sers_analysis.utils import format_column_label
from state import write_peak_artifacts_to_state
from tabs import (
    feature_analysis,
    model_consistency,
    peak_diagnostics,
    sensor_assessment,
    serotype_classification,
    spectra_viewer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="SERS Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
header_col, btn_col = st.sidebar.columns([3, 1])
with header_col:
    st.markdown("# 📁 Data Loading")
with btn_col:
    st.button("Reload Data", type="primary", on_click=clear_app_data)
uploaded = st.sidebar.file_uploader(
    "Upload Excel (.xlsx) files",
    type=["xlsx", "xls"],
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.get(UPLOADER_RESET_KEY, 'default')}",
)
loaded_bundle = None
if uploaded:
    files_data = tuple((f.name, f.getvalue()) for f in uploaded)
    loaded_bundle = load_from_uploaded(files_data)
    logger.info(
        "Loaded %d files: wide_df shape %s, tidy_df shape %s",
        len(uploaded),
        getattr(loaded_bundle.wide_df, "shape", None),
        getattr(loaded_bundle.tidy_df, "shape", None),
    )

if loaded_bundle is None or loaded_bundle.tidy_df.empty:
    logger.warning("No data loaded: tidy_df is empty or None")
    st.info("Load data using the sidebar: upload Excel (.xlsx) files.")
    st.stop()

st.sidebar.success(
    "Loaded "
    f"**{len(uploaded)}** files, **{len(loaded_bundle.wide_df)}** samples "
    f"({len(loaded_bundle.tidy_df)} tidy rows)."
)
st.sidebar.markdown(section_divider(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Raman shift trimming and peaks per serotype
# ---------------------------------------------------------------------------
min_shift, max_shift, n_peaks, n_peaks_by_serotype = render_raman_and_peaks_sidebar(
    st.sidebar, loaded_bundle.wide_df
)
derived_bundle = build_cached_derived_bundle(
    loaded_bundle,
    min_shift=min_shift,
    max_shift=max_shift,
    n_peaks=int(n_peaks),
    n_peaks_by_serotype_items=(
        tuple(sorted(n_peaks_by_serotype.items())) if n_peaks_by_serotype else ()
    ),
)
write_peak_artifacts_to_state(derived_bundle.peak_artifacts)
logger.info(
    "Raman shift trimmed: min=%s, max=%s; wide_df %d rows",
    min_shift,
    max_shift,
    len(derived_bundle.wide_df),
)
if derived_bundle.peak_artifacts.peak_infos_by_serotype:
    logger.info(
        "Peak extraction: %d serotypes, peak_cols=%s",
        len(derived_bundle.peak_artifacts.peak_infos_by_serotype),
        len(
            next(
                iter(derived_bundle.peak_artifacts.peak_infos_by_serotype.values()),
                [],
            )
        ),
    )
else:
    logger.info("Peak extraction: no serotype-specific peaks, using n_peaks=%d", n_peaks)

st.sidebar.markdown(section_divider(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Render Filter UI -> Apply Filters (dynamic from metadata columns)
# ---------------------------------------------------------------------------
filter_catalog = build_filter_catalog(
    derived_bundle.tidy_df,
    main_filter_count=MAIN_FILTER_COUNT,
)

render_main_filter_header(st.sidebar, list(filter_catalog.filter_columns))
st.sidebar.markdown(_TITLE_TO_FILTER_DIVIDER, unsafe_allow_html=True)

filter_state: dict[str, FilterSelection] = {}

for i, col in enumerate(filter_catalog.main_columns):
    if i > 0:
        st.sidebar.markdown(_FILTER_DIVIDER, unsafe_allow_html=True)
    opts_all = compute_filter_options(
        derived_bundle.tidy_df,
        filter_catalog.filter_columns,
        filter_state,
    )
    help_text = "Binned concentration." if col == "concentration_group" else ""
    selected, exclude = _render_filter(
        col,
        format_column_label(col),
        opts_all[col],
        [],
        False,
        st.sidebar,
        help_text=help_text,
        reset_button_key=f"reset_{col}",
    )
    filter_state[col] = FilterSelection(
        selected_values=tuple(str(value) for value in selected),
        exclude=exclude,
    )

with st.sidebar.expander("More Filters", expanded=False):
    for i, col in enumerate(filter_catalog.more_columns):
        if i > 0:
            st.markdown(_FILTER_DIVIDER, unsafe_allow_html=True)
        opts_all = compute_filter_options(
            derived_bundle.tidy_df,
            filter_catalog.filter_columns,
            filter_state,
        )
        help_text = "Leave empty for no filter." if col == "filename" else ""
        selected, exclude = _render_filter(
            col,
            format_column_label(col),
            opts_all[col],
            [],
            False,
            st,
            help_text=help_text,
            reset_button_key=f"reset_more_{col}",
        )
        filter_state[col] = FilterSelection(
            selected_values=tuple(str(value) for value in selected),
            exclude=exclude,
        )

filtered_bundle = apply_cached_filters(
    derived_bundle,
    serialize_filter_state(filter_state),
)
logger.info(
    "Filters applied: %d spectrum traces, %d samples (from %d tidy, %d features)",
    filtered_bundle.n_unique_spectra,
    len(filtered_bundle.filtered_features_df),
    len(derived_bundle.tidy_df),
    len(derived_bundle.features_df),
)

# ---------------------------------------------------------------------------
# 3. Main: Summary and Tabs
# ---------------------------------------------------------------------------
st.caption(
    f"Filtered to **{filtered_bundle.n_unique_spectra}** spectrum traces, "
    f"**{len(filtered_bundle.filtered_features_df)}** "
    "samples for feature analysis"
)

if filtered_bundle.filtered_tidy_df.empty:
    logger.warning("No data matches selected filters")
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
    spectra_viewer.render(filtered_bundle.filtered_tidy_df)

with tab_peak_diag:
    peak_diagnostics.render(
        filtered_bundle.filtered_features_df,
        derived_bundle.wide_df,
        derived_bundle.peak_artifacts,
    )

with tab_stats:
    feature_analysis.render(
        filtered_bundle.filtered_features_df,
        derived_bundle.peak_artifacts,
    )

with tab_assessment:
    sensor_assessment.render(
        filtered_bundle.filtered_features_df,
        derived_bundle.peak_artifacts,
    )

with tab_model_consistency:
    model_consistency.render(
        filtered_bundle.filtered_features_df,
        derived_bundle.peak_artifacts,
    )

with tab_phase2:
    serotype_classification.render(
        filtered_bundle.filtered_features_df,
        derived_bundle.peak_artifacts,
    )
