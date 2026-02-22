"""
Self-Serve SERS Data Explorer — Streamlit UI for loading, filtering, and plotting SERS spectra.

Collaborators can load datasets, filter dynamically, and generate plots without writing code.
"""

import tempfile
from pathlib import Path

import streamlit as st

from sensd_sers_analysis.data.io import load_sers_data, wide_to_tidy
from sensd_sers_analysis.visualization.plots import plot_spectra

st.set_page_config(
    page_title="SERS Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def _load_and_convert_from_paths(paths: tuple[str, ...]) -> tuple:
    """
    Load SERS data from file paths and convert to tidy format.

    Returns:
        Tuple of (wide_df, tidy_df). Empty DataFrames if loading fails.
    """
    if not paths:
        import pandas as pd

        return pd.DataFrame(), pd.DataFrame()

    wide = load_sers_data(list(paths))
    if wide.empty:
        return wide, wide

    tidy = wide_to_tidy(wide)
    return wide, tidy


@st.cache_data
def _load_and_convert_from_uploaded(
    _files_data: tuple[tuple[str, bytes], ...],
) -> tuple:
    """
    Load SERS data from uploaded file bytes and convert to tidy format.

    Args:
        _files_data: Tuple of (filename, file_bytes) per uploaded file.
                     Leading underscore to exclude from Streamlit's cache key display.

    Returns:
        Tuple of (wide_df, tidy_df). Empty DataFrames if loading fails.
    """
    import pandas as pd

    if not _files_data:
        return pd.DataFrame(), pd.DataFrame()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        paths = []
        for name, content in _files_data:
            p = tmp_path / name
            p.write_bytes(content)
            paths.append(p)
        wide = load_sers_data(paths)
        if wide.empty:
            return wide, wide
        tidy = wide_to_tidy(wide)
        return wide, tidy


# ---------------------------------------------------------------------------
# Sidebar: Data Loading
# ---------------------------------------------------------------------------
st.sidebar.header("📁 Data Loading")
load_mode = st.sidebar.radio(
    "Data source",
    options=["Directory path", "Upload files"],
    horizontal=True,
)

wide_df = None
tidy_df = None

if load_mode == "Directory path":
    data_path = st.sidebar.text_input(
        "Directory or file path",
        value="",
        placeholder="/path/to/data or /path/to/file.xlsx",
        help="Enter an absolute path to a folder (scans *.xlsx) or a single Excel file.",
    )
    if data_path.strip():
        path = Path(data_path.strip())
        if path.exists():
            paths_tuple = (str(path),)
            wide_df, tidy_df = _load_and_convert_from_paths(paths_tuple)
        else:
            st.sidebar.warning("Path does not exist. Please enter a valid path.")

else:  # Upload files
    uploaded = st.sidebar.file_uploader(
        "Upload Excel (.xlsx) files",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )
    if uploaded:
        files_data = tuple((f.name, f.getvalue()) for f in uploaded)
        wide_df, tidy_df = _load_and_convert_from_uploaded(files_data)

# ---------------------------------------------------------------------------
# Main: Show data status and filters
# ---------------------------------------------------------------------------
if tidy_df is None or tidy_df.empty:
    st.info(
        "👆 Load data using the sidebar: enter a directory path or upload Excel files."
    )
    st.stop()

st.sidebar.success(f"Loaded **{len(wide_df)}** samples, **{len(tidy_df)}** tidy rows.")

# ---------------------------------------------------------------------------
# Sidebar: Cascading filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Filters")
st.sidebar.caption("Filter the dataset before plotting. Filters cascade.")

# Build filtered DataFrame step by step for cascade
filtered = tidy_df.copy()

# 1. Serotype
serotype_options = sorted(filtered["serotype"].dropna().unique().astype(str))
serotype_selected = st.sidebar.multiselect(
    "Serotype",
    options=serotype_options,
    default=[],
    help="Leave empty to include all.",
)
if serotype_selected:
    filtered = filtered[filtered["serotype"].astype(str).isin(serotype_selected)]

# 2. Concentration (cascades from serotype)
conc_options = sorted(filtered["concentration"].dropna().unique())
conc_selected = st.sidebar.multiselect(
    "Concentration",
    options=conc_options,
    default=[],
    help="Leave empty to include all.",
)
if conc_selected:
    filtered = filtered[filtered["concentration"].isin(conc_selected)]

# 3. Filename (cascades from serotype + concentration)
filename_options = sorted(filtered["filename"].dropna().unique().astype(str))
filename_selected = st.sidebar.multiselect(
    "Filename",
    options=filename_options,
    default=[],
    help="Leave empty to include all.",
)
if filename_selected:
    filtered = filtered[filtered["filename"].astype(str).isin(filename_selected)]

# 4. Signal index (cascades from above)
sig_options = sorted(filtered["signal_index"].dropna().unique())
sig_selected = st.sidebar.multiselect(
    "Signal index",
    options=sig_options,
    default=[],
    help="Leave empty to include all.",
)
if sig_selected:
    filtered = filtered[filtered["signal_index"].isin(sig_selected)]

# ---------------------------------------------------------------------------
# Sidebar: Plot configuration
# ---------------------------------------------------------------------------
st.sidebar.header("📈 Plot Options")

# Hue and style column choices
meta_cols_for_plot = [
    "serotype",
    "concentration",
    "sensor_id",
    "test_id",
    "connection_id",
    "filename",
]
available_hue = [c for c in meta_cols_for_plot if c in filtered.columns]
available_style = [c for c in meta_cols_for_plot if c in filtered.columns]

hue_choice = st.sidebar.selectbox(
    "Hue (color grouping)",
    options=["None"] + available_hue,
    index=0,
)
style_choice = st.sidebar.selectbox(
    "Style (line style grouping)",
    options=["None"] + available_style,
    index=0,
)
show_variance = st.sidebar.checkbox(
    "Show variance band",
    value=False,
    help="Mean ± error band instead of individual lines.",
)

hue_col = None if hue_choice == "None" else hue_choice
style_col = None if style_choice == "None" else style_choice

# ---------------------------------------------------------------------------
# Main: Summary and plot
# ---------------------------------------------------------------------------
n_filtered = len(filtered.drop_duplicates(subset=["filename", "signal_index"]))
st.caption(
    f"Filtered to **{n_filtered}** spectrum traces (from {len(filtered)} tidy rows)"
)

if filtered.empty:
    st.warning("No data matches the selected filters. Adjust filters and try again.")
    st.stop()

try:
    fig = plot_spectra(
        filtered,
        hue=hue_col,
        style=style_col,
        show_variance=show_variance,
    )
    st.pyplot(fig, use_container_width=True)
except ValueError as e:
    st.error(f"Plot error: {e}")
    st.caption(
        "Ensure filtered data has required columns: raman_shift, intensity, filename, signal_index."
    )
