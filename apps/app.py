"""
Self-Serve SERS Data Explorer — Streamlit UI for loading, filtering, and plotting SERS spectra.

Collaborators can load datasets, filter dynamically, and generate plots without writing code.
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from sensd_sers_analysis.constants import CONCENTRATION_GROUP_ORDER
from sensd_sers_analysis.data.io import load_sers_data, wide_to_tidy
from sensd_sers_analysis.processing.features import extract_basic_features
from sensd_sers_analysis.visualization.plots import plot_spectra
from sensd_sers_analysis.visualization.stats import plot_feature_distribution

st.set_page_config(
    page_title="SERS Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data preprocessing — concentration binning and date normalization
# ---------------------------------------------------------------------------


def preprocess_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add concentration_group (binned) and normalize date for any DataFrame.

    Works on wide or tidy format. For wide DataFrames, concentration may be
    a list per row (one per signal); uses signal_index to pick the scalar.

    Binning rules: 0 CFU (conc < 0.1), 1 CFU (0.1–5), 10 CFU (5–50),
    100 CFU (50–500), 1000 CFU (≥500).
    Date is normalized to YYYY-MM-DD string format.
    """
    out = df.copy()
    if "concentration" in out.columns:
        conc_raw = out["concentration"]
        conc_vals = []
        for i in range(len(out)):
            c = conc_raw.iloc[i]
            if isinstance(c, (list, tuple)) and len(c) > 0:
                si = out["signal_index"].iloc[i] if "signal_index" in out.columns else 0
                idx = int(si) if pd.notna(si) else 0
                c = c[min(idx, len(c) - 1)]
            conc_vals.append(c)
        conc = pd.Series(pd.to_numeric(conc_vals, errors="coerce"), index=out.index)
        valid = conc.notna()
        cat_dtype = pd.CategoricalDtype(
            categories=CONCENTRATION_GROUP_ORDER, ordered=True
        )
        out["concentration_group"] = pd.Categorical(
            ["Unknown"] * len(out), categories=CONCENTRATION_GROUP_ORDER, ordered=True
        )
        if valid.any():
            bins = [-float("inf"), 0.1, 5, 50, 500, float("inf")]
            labels = ["0 CFU", "1 CFU", "10 CFU", "100 CFU", "1000 CFU"]
            binned = pd.cut(
                conc[valid], bins=bins, labels=labels, right=False, include_lowest=True
            )
            out.loc[valid, "concentration_group"] = pd.Categorical(
                binned.astype(str), dtype=cat_dtype
            )
    else:
        out["concentration_group"] = pd.Categorical(
            ["Unknown"] * len(out),
            categories=CONCENTRATION_GROUP_ORDER,
            ordered=True,
        )
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["date"] = out["date"].dt.strftime("%Y-%m-%d").fillna("").astype(str)
    return out


def preprocess_tidy_data(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add concentration_group (binned) and normalize date column for tidy DataFrame.

    Binning rules: 0 CFU (conc < 0.1), 1 CFU (0.1–5), 10 CFU (5–50),
    100 CFU (50–500), 1000 CFU (≥500).
    Date is normalized to YYYY-MM-DD string format.
    """
    return preprocess_metadata(tidy_df)


# ---------------------------------------------------------------------------
# Filter logic (controller) — no DataFrame copy, uses boolean masking
# ---------------------------------------------------------------------------

FLAT_OPTIONS_THRESHOLD = 50  # Use pills/flat UI when options <= this; else dropdown


def _render_filter(
    label: str,
    options: list,
    default: list,
    exclude_default: bool,
    container,
    help_text: str = "",
    label_visibility: str = "collapsed",
) -> tuple[list, bool]:
    """
    Render a filter with flat UI (pills) for small option sets, dropdown for large.
    Label and Exclude toggle share one horizontal row. Returns (selected_list, exclude_bool).
    Use label_visibility="collapsed" when a markdown title is shown above.
    """
    use_flat = len(options) <= FLAT_OPTIONS_THRESHOLD and len(options) > 0
    if not options:
        return [], exclude_default
    c1, c2 = container.columns([3, 1])
    with c1:
        if use_flat:
            selected = st.pills(
                label,
                options=options,
                default=default,
                selection_mode="multi",
                key=label,
                label_visibility=label_visibility,
            )
        else:
            selected = st.multiselect(
                label,
                options=options,
                default=default,
                help=help_text or "Leave empty to include all.",
                key=label,
                label_visibility=label_visibility,
            )
    with c2:
        exclude = st.toggle(
            "Exclude",
            value=exclude_default,
            key=f"{label}_exclude",
            help="Exclude selected instead of include only.",
        )
    return selected, exclude


def _filter_mask(
    df: pd.DataFrame,
    col: str,
    selected: list | None,
    exclude: bool,
) -> pd.Series:
    """Apply include or exclude logic for one filter dimension."""
    if not selected or col not in df.columns:
        return pd.Series(True, index=df.index)
    in_set = df[col].astype(str).isin(selected)
    return ~in_set if exclude else in_set


def get_filter_options(
    df: pd.DataFrame,
    *,
    serotype_selected: list,
    serotype_exclude: bool,
    concentration_group_selected: list,
    concentration_group_exclude: bool,
    date_selected: list,
    date_exclude: bool,
    sensor_id_selected: list,
    sensor_id_exclude: bool,
    test_id_selected: list,
    test_id_exclude: bool,
    sensor_model_selected: list,
    sensor_model_exclude: bool,
    operator_selected: list,
    operator_exclude: bool,
    filename_selected: list,
    filename_exclude: bool,
    connection_id_selected: list,
    connection_id_exclude: bool,
) -> dict[str, list]:
    """
    Compute cascading filter options. Order: Serotype, Conc, Date, Sensor ID, Test ID,
    then Sensor Model, Operator, Filename, Connection ID.
    """
    mask = pd.Series(True, index=df.index)

    def _str_opts(col: str) -> list:
        if col not in df.columns:
            return []
        series = df.loc[mask, col].dropna().astype(str)
        raw_vals = [v for v in series.unique() if v != ""]
        if col == "concentration_group" and raw_vals:
            ordered = [v for v in CONCENTRATION_GROUP_ORDER if v in raw_vals]
            extra = [v for v in raw_vals if v not in CONCENTRATION_GROUP_ORDER]
            return ordered + sorted(extra)
        return sorted(raw_vals)

    serotype_opts = _str_opts("serotype")
    mask = mask & _filter_mask(df, "serotype", serotype_selected, serotype_exclude)

    conc_grp_opts = _str_opts("concentration_group")
    mask = mask & _filter_mask(
        df,
        "concentration_group",
        concentration_group_selected,
        concentration_group_exclude,
    )

    date_opts = _str_opts("date")
    mask = mask & _filter_mask(df, "date", date_selected, date_exclude)

    sensor_id_opts = _str_opts("sensor_id")
    mask = mask & _filter_mask(df, "sensor_id", sensor_id_selected, sensor_id_exclude)

    test_id_opts = _str_opts("test_id")
    mask = mask & _filter_mask(df, "test_id", test_id_selected, test_id_exclude)

    sensor_model_opts = _str_opts("sensor_model")
    mask = mask & _filter_mask(
        df, "sensor_model", sensor_model_selected, sensor_model_exclude
    )

    operator_opts = _str_opts("operator")
    mask = mask & _filter_mask(df, "operator", operator_selected, operator_exclude)

    filename_opts = _str_opts("filename")
    mask = mask & _filter_mask(df, "filename", filename_selected, filename_exclude)

    connection_id_opts = _str_opts("connection_id")
    mask = mask & _filter_mask(
        df, "connection_id", connection_id_selected, connection_id_exclude
    )

    return {
        "serotype": serotype_opts,
        "concentration_group": conc_grp_opts,
        "date": date_opts,
        "sensor_id": sensor_id_opts,
        "test_id": test_id_opts,
        "sensor_model": sensor_model_opts,
        "operator": operator_opts,
        "filename": filename_opts,
        "connection_id": connection_id_opts,
    }


def filter_sers_data(
    df: pd.DataFrame,
    *,
    serotype_selected: list | None = None,
    serotype_exclude: bool = False,
    concentration_group_selected: list | None = None,
    concentration_group_exclude: bool = False,
    sensor_model_selected: list | None = None,
    sensor_model_exclude: bool = False,
    date_selected: list | None = None,
    date_exclude: bool = False,
    operator_selected: list | None = None,
    operator_exclude: bool = False,
    filename_selected: list | None = None,
    filename_exclude: bool = False,
    sensor_id_selected: list | None = None,
    sensor_id_exclude: bool = False,
    test_id_selected: list | None = None,
    test_id_exclude: bool = False,
    connection_id_selected: list | None = None,
    connection_id_exclude: bool = False,
    signal_index_selected: list | None = None,
    signal_index_exclude: bool = False,
) -> pd.DataFrame:
    """
    Apply cascading filters via a single compound boolean mask. No DataFrame copy.
    Exclude mode inverts the logic: exclude selected items instead of include only.

    Returns:
        Filtered DataFrame (single slice at the end).
    """
    mask = pd.Series(True, index=df.index)
    mask = mask & _filter_mask(df, "serotype", serotype_selected, serotype_exclude)
    mask = mask & _filter_mask(
        df,
        "concentration_group",
        concentration_group_selected,
        concentration_group_exclude,
    )
    mask = mask & _filter_mask(
        df, "sensor_model", sensor_model_selected, sensor_model_exclude
    )
    mask = mask & _filter_mask(df, "date", date_selected, date_exclude)
    mask = mask & _filter_mask(df, "operator", operator_selected, operator_exclude)
    mask = mask & _filter_mask(df, "filename", filename_selected, filename_exclude)
    mask = mask & _filter_mask(df, "sensor_id", sensor_id_selected, sensor_id_exclude)
    mask = mask & _filter_mask(df, "test_id", test_id_selected, test_id_exclude)
    mask = mask & _filter_mask(
        df, "connection_id", connection_id_selected, connection_id_exclude
    )
    mask = mask & _filter_mask(
        df, "signal_index", signal_index_selected, signal_index_exclude
    )
    return df.loc[mask]


# ---------------------------------------------------------------------------
# Data loading (unchanged signatures)
# ---------------------------------------------------------------------------


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
# 1. Load Data
# ---------------------------------------------------------------------------
st.sidebar.header("📁 Data Loading")
load_mode = st.sidebar.radio(
    "Data source",
    options=["Upload files", "Directory path"],
    index=0,
    horizontal=True,
)

wide_df = None
tidy_df = None

if load_mode == "Upload files":
    uploaded = st.sidebar.file_uploader(
        "Upload Excel (.xlsx) files",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )
    if uploaded:
        files_data = tuple((f.name, f.getvalue()) for f in uploaded)
        wide_df, tidy_df = _load_and_convert_from_uploaded(files_data)
else:
    data_path = st.sidebar.text_input(
        "Directory or file path",
        value="",
        placeholder="/path/to/data or /path/to/file.xlsx",
        help="Enter an absolute path to a folder (scans *.xlsx) or a single Excel file.",
    )
    if data_path.strip():
        path = Path(data_path.strip())
        if path.exists():
            wide_df, tidy_df = _load_and_convert_from_paths((str(path),))
        else:
            st.sidebar.warning("Path does not exist. Please enter a valid path.")

if tidy_df is None or tidy_df.empty:
    st.info(
        "👆 Load data using the sidebar: enter a directory path or upload Excel files."
    )
    st.stop()

tidy_df = preprocess_tidy_data(tidy_df)
wide_df = preprocess_metadata(wide_df)
features_df = extract_basic_features(wide_df)
st.sidebar.success(f"Loaded **{len(wide_df)}** samples, **{len(tidy_df)}** tidy rows.")

# ---------------------------------------------------------------------------
# 2. Render Filter UI -> Apply Filters (Faceted Navigation)
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Filters")
st.sidebar.caption("Filter the dataset before plotting. Use Exclude for fast exclude.")


def _empty_list() -> list:
    return []


def _opts(
    serotype_selected,
    serotype_exclude,
    concentration_group_selected,
    concentration_group_exclude,
    date_selected,
    date_exclude,
    sensor_id_selected,
    sensor_id_exclude,
    test_id_selected,
    test_id_exclude,
    sensor_model_selected,
    sensor_model_exclude,
    operator_selected,
    operator_exclude,
    filename_selected,
    filename_exclude,
    connection_id_selected,
    connection_id_exclude,
):
    return get_filter_options(
        tidy_df,
        serotype_selected=serotype_selected,
        serotype_exclude=serotype_exclude,
        concentration_group_selected=concentration_group_selected,
        concentration_group_exclude=concentration_group_exclude,
        date_selected=date_selected,
        date_exclude=date_exclude,
        sensor_id_selected=sensor_id_selected,
        sensor_id_exclude=sensor_id_exclude,
        test_id_selected=test_id_selected,
        test_id_exclude=test_id_exclude,
        sensor_model_selected=sensor_model_selected,
        sensor_model_exclude=sensor_model_exclude,
        operator_selected=operator_selected,
        operator_exclude=operator_exclude,
        filename_selected=filename_selected,
        filename_exclude=filename_exclude,
        connection_id_selected=connection_id_selected,
        connection_id_exclude=connection_id_exclude,
    )


# Main filters: Serotype, Concentration Group, Date, Sensor ID, Test ID
opts = _opts(
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
)
st.sidebar.markdown("### Serotype")
serotype_selected, serotype_exclude = _render_filter(
    "Serotype",
    opts["serotype"],
    [],
    False,
    st.sidebar,
)

opts = _opts(
    serotype_selected,
    serotype_exclude,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
)
st.sidebar.markdown("### Concentration Group")
concentration_group_selected, concentration_group_exclude = _render_filter(
    "Concentration Group",
    opts["concentration_group"],
    [],
    False,
    st.sidebar,
    help_text="Binned concentration.",
)

opts = _opts(
    serotype_selected,
    serotype_exclude,
    concentration_group_selected,
    concentration_group_exclude,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
)
st.sidebar.markdown("### Date")
date_selected, date_exclude = _render_filter(
    "Date",
    opts["date"],
    [],
    False,
    st.sidebar,
)

opts = _opts(
    serotype_selected,
    serotype_exclude,
    concentration_group_selected,
    concentration_group_exclude,
    date_selected,
    date_exclude,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
)
st.sidebar.markdown("### Sensor ID")
sensor_id_selected, sensor_id_exclude = _render_filter(
    "Sensor ID",
    opts["sensor_id"],
    [],
    False,
    st.sidebar,
)

opts = _opts(
    serotype_selected,
    serotype_exclude,
    concentration_group_selected,
    concentration_group_exclude,
    date_selected,
    date_exclude,
    sensor_id_selected,
    sensor_id_exclude,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
    _empty_list(),
    False,
)
st.sidebar.markdown("### Test ID")
test_id_selected, test_id_exclude = _render_filter(
    "Test ID",
    opts["test_id"],
    [],
    False,
    st.sidebar,
)

with st.sidebar.expander("More Filters", expanded=False):
    opts = _opts(
        serotype_selected,
        serotype_exclude,
        concentration_group_selected,
        concentration_group_exclude,
        date_selected,
        date_exclude,
        sensor_id_selected,
        sensor_id_exclude,
        test_id_selected,
        test_id_exclude,
        _empty_list(),
        False,
        _empty_list(),
        False,
        _empty_list(),
        False,
        _empty_list(),
        False,
    )
    st.markdown("### Sensor Model")
    sensor_model_selected, sensor_model_exclude = _render_filter(
        "Sensor Model",
        opts["sensor_model"],
        [],
        False,
        st,
    )
    opts = _opts(
        serotype_selected,
        serotype_exclude,
        concentration_group_selected,
        concentration_group_exclude,
        date_selected,
        date_exclude,
        sensor_id_selected,
        sensor_id_exclude,
        test_id_selected,
        test_id_exclude,
        sensor_model_selected,
        sensor_model_exclude,
        _empty_list(),
        False,
        _empty_list(),
        False,
        _empty_list(),
        False,
    )
    st.markdown("### Operator")
    operator_selected, operator_exclude = _render_filter(
        "Operator",
        opts["operator"],
        [],
        False,
        st,
    )
    opts = _opts(
        serotype_selected,
        serotype_exclude,
        concentration_group_selected,
        concentration_group_exclude,
        date_selected,
        date_exclude,
        sensor_id_selected,
        sensor_id_exclude,
        test_id_selected,
        test_id_exclude,
        sensor_model_selected,
        sensor_model_exclude,
        operator_selected,
        operator_exclude,
        _empty_list(),
        False,
        _empty_list(),
        False,
    )
    st.markdown("### Filename")
    filename_selected, filename_exclude = _render_filter(
        "Filename",
        opts["filename"],
        [],
        False,
        st,
        help_text="Leave empty for no filter.",
    )
    opts = _opts(
        serotype_selected,
        serotype_exclude,
        concentration_group_selected,
        concentration_group_exclude,
        date_selected,
        date_exclude,
        sensor_id_selected,
        sensor_id_exclude,
        test_id_selected,
        test_id_exclude,
        sensor_model_selected,
        sensor_model_exclude,
        operator_selected,
        operator_exclude,
        filename_selected,
        filename_exclude,
        _empty_list(),
        False,
    )
    st.markdown("### Connection ID")
    connection_id_selected, connection_id_exclude = _render_filter(
        "Connection ID",
        opts["connection_id"],
        [],
        False,
        st,
    )

_filter_args = dict(
    serotype_selected=serotype_selected if serotype_selected else None,
    serotype_exclude=serotype_exclude,
    concentration_group_selected=concentration_group_selected
    if concentration_group_selected
    else None,
    concentration_group_exclude=concentration_group_exclude,
    sensor_model_selected=sensor_model_selected if sensor_model_selected else None,
    sensor_model_exclude=sensor_model_exclude,
    date_selected=date_selected if date_selected else None,
    date_exclude=date_exclude,
    operator_selected=operator_selected if operator_selected else None,
    operator_exclude=operator_exclude,
    filename_selected=filename_selected if filename_selected else None,
    filename_exclude=filename_exclude,
    sensor_id_selected=sensor_id_selected if sensor_id_selected else None,
    sensor_id_exclude=sensor_id_exclude,
    test_id_selected=test_id_selected if test_id_selected else None,
    test_id_exclude=test_id_exclude,
    connection_id_selected=connection_id_selected if connection_id_selected else None,
    connection_id_exclude=connection_id_exclude,
    signal_index_selected=None,
    signal_index_exclude=False,
)
filtered = filter_sers_data(tidy_df, **_filter_args)
filtered_features = filter_sers_data(features_df, **_filter_args)

# ---------------------------------------------------------------------------
# 3. Main: Summary and Tabs (Spectral Viewer | Feature Analysis)
# ---------------------------------------------------------------------------
n_filtered = len(filtered.drop_duplicates(subset=["filename", "signal_index"]))
st.caption(
    f"Filtered to **{n_filtered}** spectrum traces, **{len(filtered_features)}** samples for feature analysis"
)

if filtered.empty:
    st.warning("No data matches the selected filters. Adjust filters and try again.")
    st.stop()

tab_spectra, tab_stats = st.tabs(["📉 Spectral Viewer", "📊 Feature Analysis"])

with tab_spectra:
    meta_cols_for_plot = [
        "concentration_group",
        "serotype",
        "sensor_model",
        "date",
        "operator",
        "concentration",
        "sensor_id",
        "test_id",
        "connection_id",
        "filename",
    ]
    available_cols = [c for c in meta_cols_for_plot if c in filtered.columns]
    hue_options = ["None"] + available_cols
    hue_default = "None"
    for preferred in ("concentration_group", "serotype"):
        if preferred in available_cols:
            hue_default = preferred
            break

    st.markdown("#### Spectra plot options")
    c_hue, c_style, c_var, c_h = st.columns(4)
    with c_hue:
        hue_choice = st.radio(
            "Hue _(color)_",
            options=hue_options,
            index=hue_options.index(hue_default),
            horizontal=False,
            key="hue_radio",
        )
    with c_style:
        style_choice = st.radio(
            "Style _(line)_",
            options=["None"] + available_cols,
            index=0,
            horizontal=False,
            key="style_radio",
        )
    with c_var:
        VARIANCE_OPTIONS = [
            ("Individual Lines", False, "sd"),
            ("±1 SD", True, "sd"),
            ("±1 SE", True, "se"),
            ("95% CI", True, ("ci", 95)),
        ]
        variance_labels = [v[0] for v in VARIANCE_OPTIONS]
        variance_choice = st.radio(
            "Display",
            options=variance_labels,
            index=0,
            horizontal=False,
            key="variance_radio",
        )
    with c_h:
        plot_height = st.slider(
            "Height (in)",
            min_value=4,
            max_value=12,
            value=6,
            step=1,
            key="plot_height_slider",
        )

    _vo = VARIANCE_OPTIONS[variance_labels.index(variance_choice)]
    show_variance = _vo[1]
    errorbar = _vo[2]
    hue_col = None if hue_choice == "None" else hue_choice
    style_col = None if style_choice == "None" else style_choice

    try:
        fig = plot_spectra(
            filtered,
            hue=hue_col,
            style=style_col,
            show_variance=show_variance,
            errorbar=errorbar,
            figsize=(14, plot_height),
        )
        st.pyplot(fig, use_container_width=True)
    except ValueError as e:
        st.error(f"Plot error: {e}")
        st.caption(
            "Ensure filtered data has required columns: raman_shift, intensity, "
            "filename, signal_index."
        )

with tab_stats:
    feat_cols = ["max_intensity", "mean_intensity", "integral_area"]
    all_feat_nan = all(
        filtered_features[c].isna().all()
        for c in feat_cols
        if c in filtered_features.columns
    )
    if all_feat_nan:
        st.warning(
            "All extracted features are NaN. This usually means the loaded data "
            "lacks Raman intensity columns (rs_*) or they contain no valid "
            "numeric values. Ensure your Excel files use the expected embedded "
            "format with Raman shift columns."
        )
    else:
        st.markdown("#### Feature distribution options")
        x_opts = [c for c in filtered_features.columns if c not in feat_cols]
        hue_opts = ["None"] + x_opts

        x_default = (
            "concentration_group"
            if "concentration_group" in x_opts
            else (x_opts[0] if x_opts else None)
        )
        x_default_idx = (
            x_opts.index(x_default) if x_default and x_default in x_opts else 0
        )

        c_feat, c_x, c_hue_s, c_type = st.columns(4)
        with c_feat:
            feature_col = st.selectbox(
                "Feature (Y-axis)",
                options=feat_cols,
                index=2,
                key="stats_feature",
            )
        with c_x:
            x_col = st.selectbox(
                "X-axis",
                options=x_opts if x_opts else ["(no grouping)"],
                index=min(x_default_idx, len(x_opts) - 1) if x_opts else 0,
                key="stats_x",
            )
        with c_hue_s:
            hue_default_s = "serotype" if "serotype" in hue_opts else "None"
            hue_col_s = st.selectbox(
                "Hue",
                options=hue_opts,
                index=hue_opts.index(hue_default_s),
                key="stats_hue",
            )
        with c_type:
            plot_type = st.radio(
                "Plot type",
                options=["box", "violin"],
                index=0,
                horizontal=True,
                key="stats_plot_type",
            )

        if filtered_features.empty:
            st.warning("No samples match the selected filters for feature analysis.")
        else:
            x_val = (
                None
                if (
                    not x_col
                    or x_col == "(no grouping)"
                    or x_col not in filtered_features.columns
                )
                else x_col
            )
            hue_val = None if hue_col_s == "None" else hue_col_s
            try:
                fig_stats = plot_feature_distribution(
                    filtered_features,
                    feature_col,
                    x=x_val,
                    hue=hue_val,
                    plot_type=plot_type,
                )
                st.pyplot(fig_stats, use_container_width=True)
            except ValueError as e:
                st.error(f"Plot error: {e}")
