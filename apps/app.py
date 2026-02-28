"""
Self-Serve SERS Data Explorer — Streamlit UI for loading, filtering, and plotting SERS spectra.

Collaborators can load datasets, filter dynamically, and generate plots without writing code.
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from sensd_sers_analysis.data import (
    count_unique_spectra,
    load_sers_data_as_wide_and_tidy,
    wide_to_tidy,
)
from sensd_sers_analysis.processing import (
    BASIC_FEATURE_COLUMNS,
    extract_basic_features,
    filter_sers_data,
    trim_raman_shift,
    get_feature_metadata_columns,
    get_filter_options,
    get_filterable_columns,
    get_plot_hue_columns,
    pick_preferred_column,
    preprocess_metadata,
)
from sensd_sers_analysis.assessment import (
    compute_batch_variance,
    compute_degradation,
    get_consistency_summary_table,
    identify_deviating_sensors,
    prepare_degradation_data,
)
from sensd_sers_analysis.report import build_sensor_assessment_pdf
from sensd_sers_analysis.utils import (
    format_column_label,
    order_concentration_labels,
)
from sensd_sers_analysis.visualization import (
    plot_batch_boxplot,
    plot_degradation_trend,
    plot_feature_distribution,
    plot_spectra,
)

st.set_page_config(
    page_title="SERS Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Filter UI (controller uses src.processing.filters)
# ---------------------------------------------------------------------------

FLAT_OPTIONS_THRESHOLD = 50  # Use pills/flat UI when options <= this; else dropdown

# Dividers use currentColor so they adapt to light/dark theme
_SECTION_DIVIDER = (
    '<hr style="margin: 1rem 0; padding: 0; border: 0; '
    'border-top: 3px solid currentColor; opacity: 0.5;">'
)
_TITLE_TO_FILTER_DIVIDER = (
    '<hr style="margin: 0.4rem 0; padding: 0; border: 0; '
    'border-top: 1px solid currentColor; opacity: 0.25;">'
)
_FILTER_DIVIDER = (
    '<hr style="margin: 0.25rem 0; padding: 0; border: 0; '
    'border-top: 1px solid currentColor; opacity: 0.2;">'
)


def _render_filter(
    label: str,
    options: list,
    default: list,
    exclude_default: bool,
    container,
    *,
    help_text: str = "",
    label_visibility: str = "collapsed",
    reset_button_key: str | None = None,
) -> tuple[list, bool]:
    """
    Render a filter: title row [Label + Exclude] ... [Reset], then selection widget below.
    Returns (selected_list, exclude_bool).
    """
    use_flat = len(options) <= FLAT_OPTIONS_THRESHOLD and len(options) > 0
    if not options:
        return [], exclude_default

    # Title row: [Title] [Exclude] ............... [Reset]
    # Handle Reset first, before any widgets that own these session keys.
    title_cols = container.columns([4, 1])
    with title_cols[1]:
        if reset_button_key and st.button(
            "Reset",
            key=reset_button_key,
            help="Reset selection and Exclude for this filter.",
        ):
            st.session_state[label] = []
            st.session_state[f"{label}_exclude"] = False
            st.rerun()
    with title_cols[0]:
        sub = st.columns([2, 1])  # title, exclude (both left)
        with sub[0]:
            st.markdown(f"### {label}")
        with sub[1]:
            exclude = st.toggle(
                "Exclude",
                value=exclude_default,
                key=f"{label}_exclude",
                help="Exclude selected instead of include only.",
            )

    # Selection widget below (full width)
    if use_flat:
        selected = container.pills(
            label,
            options=options,
            default=default,
            selection_mode="multi",
            key=label,
            label_visibility=label_visibility,
        )
    else:
        selected = container.multiselect(
            label,
            options=options,
            default=default,
            help=help_text or "Leave empty to include all.",
            key=label,
            label_visibility=label_visibility,
        )
    return selected, exclude


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


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
    if not _files_data:
        return pd.DataFrame(), pd.DataFrame()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        paths = [str(tmp_path / name) for name, _ in _files_data]
        for (name, content), p in zip(_files_data, paths):
            Path(p).write_bytes(content)
        return load_sers_data_as_wide_and_tidy(paths)


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
    wide_df, tidy_df = _load_and_convert_from_uploaded(files_data)

if tidy_df is None or tidy_df.empty:
    st.info("👆 Load data using the sidebar: upload Excel (.xlsx) files.")
    st.stop()

tidy_df = preprocess_metadata(tidy_df)
wide_df = preprocess_metadata(wide_df)

# ---------------------------------------------------------------------------
# Raman Shift trimming (apply before feature extraction and plotting)
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### Raman Shift window")
st.sidebar.caption(
    "Trim spectra to a uniform window. Leave blank for no limit. "
    "Common default: 400–1800 cm⁻¹."
)
rs_min_str = st.sidebar.text_input(
    "Min Raman Shift (cm⁻¹)",
    value="",
    placeholder="e.g. 400",
    key="raman_shift_min",
    help="Leave blank for no lower limit.",
)
rs_max_str = st.sidebar.text_input(
    "Max Raman Shift (cm⁻¹)",
    value="",
    placeholder="e.g. 1800",
    key="raman_shift_max",
    help="Leave blank for no upper limit.",
)


def _parse_raman_shift_bound(s: str) -> float | None:
    """Parse user input to float; return None if blank or invalid."""
    if not s or not str(s).strip():
        return None
    try:
        return float(str(s).strip())
    except ValueError:
        return None


min_shift = _parse_raman_shift_bound(rs_min_str)
max_shift = _parse_raman_shift_bound(rs_max_str)
wide_df = trim_raman_shift(wide_df, min_shift=min_shift, max_shift=max_shift)
tidy_df = wide_to_tidy(wide_df)
tidy_df = preprocess_metadata(tidy_df)

features_df = extract_basic_features(wide_df)
st.sidebar.success(
    f"Loaded **{len(uploaded)}** files, **{len(wide_df)}** samples ({len(tidy_df)} tidy rows)."
)

st.sidebar.markdown(_SECTION_DIVIDER, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Render Filter UI -> Apply Filters (dynamic from metadata columns)
# ---------------------------------------------------------------------------
filter_columns = get_filterable_columns(tidy_df)

filter_title_cols = st.sidebar.columns([4, 1])  # Filters left, Reset all right
with filter_title_cols[0]:
    st.markdown("# 🔍 Filters")
with filter_title_cols[1]:
    if st.button(
        "Reset all",
        key="reset_all_filters",
        help="Reset all filter selections and Exclude toggles.",
    ):
        for col in filter_columns:
            label = format_column_label(col)
            st.session_state[label] = []
            st.session_state[f"{label}_exclude"] = False
        st.rerun()

st.sidebar.markdown(_TITLE_TO_FILTER_DIVIDER, unsafe_allow_html=True)

MAIN_FILTER_COUNT = 5  # Serotype, Concentration Group, Date, Sensor ID, Test ID
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
# 3. Main: Summary and Tabs (Spectral Viewer | Feature Analysis)
# ---------------------------------------------------------------------------
n_filtered = count_unique_spectra(filtered)
st.caption(
    f"Filtered to **{n_filtered}** spectrum traces, **{len(filtered_features)}** samples for feature analysis"
)

if filtered.empty:
    st.warning("No data matches the selected filters. Adjust filters and try again.")
    st.stop()

tab_spectra, tab_stats, tab_assessment = st.tabs(
    ["📉 Spectral Viewer", "📊 Feature Analysis", "🔬 Sensor Assessment & Report"]
)

with tab_spectra:
    available_cols = get_plot_hue_columns(filtered)
    hue_options = ["None"] + available_cols
    hue_default = pick_preferred_column(available_cols) or "None"

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
    all_feat_nan = all(
        filtered_features[c].isna().all()
        for c in BASIC_FEATURE_COLUMNS
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
        x_opts = get_feature_metadata_columns(filtered_features)
        hue_opts = ["None"] + x_opts

        x_default = pick_preferred_column(x_opts) or (x_opts[0] if x_opts else None)
        x_default_idx = x_opts.index(x_default) if x_default in x_opts else 0

        c_feat, c_x, c_hue_s, c_type = st.columns(4)
        with c_feat:
            feature_col = st.selectbox(
                "Feature (Y-axis)",
                options=BASIC_FEATURE_COLUMNS,
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
            hue_default_s = pick_preferred_column(x_opts, ("serotype",)) or "None"
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

# ---------------------------------------------------------------------------
# 4. Sensor Assessment & Report
# ---------------------------------------------------------------------------
# Fixed group cols for apples-to-apples: variance/CV only within same
# sensor + serotype + concentration.
ASSESSMENT_GROUP_COLS = ["sensor_id", "serotype", "concentration_group"]

with tab_assessment:
    feat_cols_avail = [
        c for c in BASIC_FEATURE_COLUMNS if c in filtered_features.columns
    ]
    has_serotype = "serotype" in filtered_features.columns
    has_conc_group = "concentration_group" in filtered_features.columns

    if not feat_cols_avail:
        st.warning(
            "No feature columns available. Load data with Raman intensity columns "
            "and ensure filters yield samples."
        )
    elif not has_serotype or not has_conc_group:
        st.warning(
            "Assessment requires **serotype** and **concentration_group** columns. "
            "Ensure data is loaded with metadata and preprocess_metadata has run."
        )
    else:
        # ---- 1. Experimental variable control: single serotype + concentration ----
        st.markdown(
            "#### Experimental variable control\n"
            "Select a **specific serotype** and **concentration group** before running "
            "assessment. Statistics are computed only on replicates sharing these conditions."
        )
        serotype_opts = sorted(
            filtered_features["serotype"].dropna().unique().astype(str).tolist()
        ) or ["(none)"]
        conc_raw = (
            filtered_features["concentration_group"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        conc_opts = [c for c in conc_raw if c and c != "nan"]
        conc_opts = order_concentration_labels(conc_opts) if conc_opts else ["(none)"]

        a_sero, a_conc, a_feat, a_outlier = st.columns(4)
        with a_sero:
            assess_serotype = st.selectbox(
                "Serotype _(required)_",
                options=serotype_opts,
                index=0,
                key="assess_serotype",
            )
        with a_conc:
            assess_concentration = st.selectbox(
                "Concentration group _(required)_",
                options=conc_opts,
                index=0,
                key="assess_concentration",
            )
        with a_feat:
            assess_feature = st.selectbox(
                "Feature",
                options=feat_cols_avail,
                index=0,
                key="assess_feature",
            )
        with a_outlier:
            outlier_method = st.radio(
                "Outlier method",
                options=["iqr", "zscore"],
                index=0,
                horizontal=True,
                key="assess_outlier",
            )

        # Filter to selected conditions (apples-to-apples)
        _sero_valid = assess_serotype and assess_serotype != "(none)"
        _conc_valid = assess_concentration and assess_concentration != "(none)"
        if _sero_valid and _conc_valid:
            assessment_df = filtered_features[
                (filtered_features["serotype"].astype(str) == assess_serotype)
                & (
                    filtered_features["concentration_group"].astype(str)
                    == assess_concentration
                )
            ].copy()
        else:
            assessment_df = pd.DataFrame()

        if assessment_df.empty and (_sero_valid and _conc_valid):
            st.warning(
                f"No samples for serotype={assess_serotype}, concentration={assess_concentration}. "
                "Adjust filters or selection."
            )
        elif not _sero_valid or not _conc_valid:
            st.info(
                "Select a specific serotype and concentration group above to run assessment."
            )
        else:
            # Use fixed group cols for consistency (only include cols that exist)
            consistency_group_cols = [
                c for c in ASSESSMENT_GROUP_COLS if c in assessment_df.columns
            ]
            if not consistency_group_cols:
                consistency_group_cols = (
                    ["sensor_id"] if "sensor_id" in assessment_df.columns else None
                )

            # ---- 2. Consistency (CV: raw vs. filtered) ----
            st.markdown("##### Consistency (CV: raw vs. filtered)")
            st.caption(
                f"Within serotype={assess_serotype}, concentration={assess_concentration}. "
                "Grouped by sensor_id, serotype, concentration_group."
            )
            try:
                consistency_tbl = get_consistency_summary_table(
                    assessment_df,
                    feature_cols=[assess_feature],
                    group_cols=consistency_group_cols,
                    outlier_method=outlier_method,
                )
                if not consistency_tbl.empty:
                    st.dataframe(
                        consistency_tbl, use_container_width=True, hide_index=True
                    )
            except ValueError as e:
                st.error(f"Consistency error: {e}")

            # ---- 3. Degradation (test_id / date as temporal axis) ----
            st.markdown("##### Degradation trend")
            st.caption(
                "Feature vs. test sequence (test_id or date ordered). "
                "Negative slope indicates degradation."
            )
            try:
                df_deg = prepare_degradation_data(
                    assessment_df, assess_feature, test_col="test_id", date_col="date"
                )
                if df_deg.empty or len(df_deg) < 2:
                    st.info(
                        "Insufficient temporal data (need test_id or date with ≥2 tests)."
                    )
                else:
                    deg_tbl = compute_degradation(
                        df_deg,
                        assess_feature,
                        "test_ordinal",
                        group_cols=["sensor_id"]
                        if "sensor_id" in df_deg.columns
                        else None,
                    )
                    if not deg_tbl.empty:
                        st.dataframe(deg_tbl, use_container_width=True, hide_index=True)

                    fig_deg = plot_degradation_trend(
                        df_deg,
                        assess_feature,
                        "test_ordinal",
                        group_col="sensor_id"
                        if "sensor_id" in df_deg.columns
                        else None,
                    )
                    st.pyplot(fig_deg, use_container_width=True)
            except ValueError as e:
                st.error(f"Degradation error: {e}")

            # ---- 4. Multi-sensor batch stability ----
            st.markdown("---")
            st.markdown("#### Multi-sensor batch stability")
            st.caption(
                f"Within serotype={assess_serotype}, concentration={assess_concentration}. "
                "Compare sensors under identical conditions."
            )
            if "sensor_id" in assessment_df.columns:
                batch_feature = st.selectbox(
                    "Feature (batch)",
                    options=feat_cols_avail,
                    index=feat_cols_avail.index(assess_feature)
                    if assess_feature in feat_cols_avail
                    else 0,
                    key="batch_feature",
                )
                try:
                    batch_tbl = compute_batch_variance(
                        assessment_df,
                        batch_feature,
                        sensor_col="sensor_id",
                        group_cols=None,  # already filtered to one serotype+concentration
                    )
                    deviating = identify_deviating_sensors(
                        batch_tbl, z_threshold=2.0, sensor_col="sensor_id"
                    )

                    st.dataframe(batch_tbl, use_container_width=True, hide_index=True)
                    if not deviating.empty:
                        st.markdown("**Deviating sensors (|z| > 2)**")
                        st.dataframe(
                            deviating, use_container_width=True, hide_index=True
                        )

                    fig_batch = plot_batch_boxplot(
                        assessment_df,
                        batch_feature,
                        sensor_col="sensor_id",
                        group_col=None,
                    )
                    st.pyplot(fig_batch, use_container_width=True)
                except ValueError as e:
                    st.error(f"Batch variance error: {e}")
            else:
                st.info(
                    "No sensor_id column; batch analysis requires sensor identifiers."
                )

            # ---- 5. PDF report (bound to user selections) ----
            st.markdown("---")
            st.markdown("#### PDF report")
            if st.button("Generate report", key="pdf_report_btn"):
                try:
                    consistency_summary = get_consistency_summary_table(
                        assessment_df,
                        group_cols=consistency_group_cols,
                        outlier_method=outlier_method,
                    )
                    df_deg_pdf = prepare_degradation_data(
                        assessment_df,
                        assess_feature,
                        test_col="test_id",
                        date_col="date",
                    )
                    deg_summary = pd.DataFrame()
                    fig_deg_pdf = None
                    if not df_deg_pdf.empty and len(df_deg_pdf) >= 2:
                        deg_summary = compute_degradation(
                            df_deg_pdf,
                            assess_feature,
                            "test_ordinal",
                            group_cols=(
                                ["sensor_id"]
                                if "sensor_id" in df_deg_pdf.columns
                                else None
                            ),
                        )
                        fig_deg_pdf = plot_degradation_trend(
                            df_deg_pdf,
                            assess_feature,
                            "test_ordinal",
                            group_col=(
                                "sensor_id"
                                if "sensor_id" in df_deg_pdf.columns
                                else None
                            ),
                        )

                    batch_tbl_pdf = pd.DataFrame()
                    fig_batch_pdf = None
                    deviating_pdf = pd.DataFrame()
                    if "sensor_id" in assessment_df.columns:
                        batch_tbl_pdf = compute_batch_variance(
                            assessment_df,
                            assess_feature,
                            sensor_col="sensor_id",
                            group_cols=None,
                        )
                        deviating_pdf = identify_deviating_sensors(
                            batch_tbl_pdf, z_threshold=2.0, sensor_col="sensor_id"
                        )
                        fig_batch_pdf = plot_batch_boxplot(
                            assessment_df,
                            assess_feature,
                            sensor_col="sensor_id",
                            group_col=None,
                        )

                    pdf_bytes = build_sensor_assessment_pdf(
                        consistency_table=consistency_summary,
                        degradation_table=deg_summary,
                        degradation_fig=fig_deg_pdf,
                        batch_variance_table=(
                            batch_tbl_pdf if not batch_tbl_pdf.empty else None
                        ),
                        batch_boxplot_fig=fig_batch_pdf,
                        deviating_sensors_table=(
                            deviating_pdf if not deviating_pdf.empty else None
                        ),
                        outlier_method=outlier_method,
                        report_title=(
                            f"SERS Sensor Assessment — {assess_serotype}, "
                            f"{assess_concentration}"
                        ),
                    )
                    st.session_state["assessment_pdf"] = pdf_bytes
                    st.success("Report generated. Click Download below.")
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

        if "assessment_pdf" in st.session_state:
            st.download_button(
                label="Download PDF",
                data=st.session_state["assessment_pdf"],
                file_name="sensor_assessment_report.pdf",
                mime="application/pdf",
                key="pdf_download",
            )
