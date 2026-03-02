"""
Self-Serve SERS Data Explorer — Streamlit UI for loading, filtering, and plotting SERS spectra.

Collaborators can load datasets, filter dynamically, and generate plots without writing code.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sensd_sers_analysis.data import (
    count_unique_spectra,
    get_raman_shift,
    get_signals_matrix,
    load_sers_data_as_wide_and_tidy,
    wide_to_tidy,
)
from sensd_sers_analysis.processing import (
    BASIC_FEATURE_COLUMNS,
    DEFAULT_GLOBAL_QA_FEATURES,
    extract_basic_features,
    extract_dynamic_peak_features,
    filter_sers_data,
    get_peak_height_columns,
    trim_raman_shift,
    get_feature_metadata_columns,
    get_filter_options,
    get_filterable_columns,
    get_plot_hue_columns,
    order_features_by_preference,
    pick_preferred_column,
    preprocess_metadata,
)
from sensd_sers_analysis.assessment import (
    compute_batch_variance,
    compute_degradation,
    fit_concentration_regression_cleaned,
    get_consistency_summary_table,
    get_global_model_consistency_qa,
    get_zero_cfu_baseline,
    identify_deviating_sensors,
    prepare_degradation_data,
)
from sensd_sers_analysis.report import build_phase1_qa_pdf, build_sensor_assessment_pdf
from sensd_sers_analysis.utils import (
    format_column_label,
    order_concentration_labels,
)
from sensd_sers_analysis.visualization import (
    plot_batch_boxplot,
    plot_concentration_regression,
    plot_degradation_trend,
    plot_feature_distribution,
    plot_macro_batch_regression,
    plot_multi_sensor_regression,
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
# Per-serotype number of peaks (sidebar)
_serotypes_from_wide = (
    sorted(wide_df["serotype"].dropna().unique().astype(str).tolist())
    if wide_df is not None and not wide_df.empty and "serotype" in wide_df.columns
    else []
)
_serotypes_from_wide = [s for s in _serotypes_from_wide if s and s != "nan"]

if _serotypes_from_wide:
    n_peaks = 6
    st.sidebar.markdown("#### Peaks per serotype")
    st.sidebar.caption(
        "Number of peaks to extract for each serotype. Different serovars "
        "may have different numbers of prominent peaks."
    )
    n_peaks_by_serotype = {}
    for _s in _serotypes_from_wide:
        n_peaks_by_serotype[_s] = int(
            st.sidebar.number_input(
                f"Peaks ({_s})",
                min_value=1,
                max_value=10,
                value=6,
                step=1,
                key=f"n_peaks_{_s}",
                help=f"Peaks for {_s}",
            )
        )
else:
    n_peaks_by_serotype = None
    n_peaks = st.sidebar.number_input(
        "Number of Peaks",
        min_value=1,
        max_value=10,
        value=6,
        step=1,
        key="n_peaks",
        help="Number of peaks (no serotype column in data).",
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

# Dynamic peak extraction (serotype-specific, 0 CFU excluded from learning)
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

tab_spectra, tab_peak_diag, tab_stats, tab_assessment, tab_model_consistency = st.tabs(
    [
        "📉 Spectral Viewer",
        "🔍 Peak Detection Diagnostics",
        "📊 Feature Analysis",
        "🔬 Sensor Assessment & Report",
        "📈 Model-Based Sensor Consistency",
    ]
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
        st.pyplot(fig, width="stretch")
    except ValueError as e:
        st.error(f"Plot error: {e}")
        st.caption(
            "Ensure filtered data has required columns: raman_shift, intensity, "
            "filename, signal_index."
        )

with tab_peak_diag:
    import matplotlib.pyplot as plt

    peak_by_sero = st.session_state.get("peak_infos_by_serotype", {})
    mean_by_sero = st.session_state.get("mean_spec_by_serotype", {})
    raman_x = st.session_state.get("raman_x", np.array([]))

    if not peak_by_sero or raman_x.size == 0:
        st.info(
            "Peak detection requires loaded data with Raman intensity columns. "
            "Adjust **Peaks per serotype** in the sidebar and ensure high-concentration "
            "samples are present (>0 CFU, serotype-specific)."
        )
    else:
        st.markdown(
            "Visual verification of dynamic peak extraction: serotype-specific "
            "anchors, search windows, and detection success rates (0 CFU excluded from "
            "learning). Each serotype uses its own peak count from the sidebar."
        )

        # ---- All serotypes: plots and tables ----
        for sel_serotype in sorted(peak_by_sero.keys()):
            peak_infos = peak_by_sero.get(sel_serotype, [])
            mean_spec = mean_by_sero.get(sel_serotype, np.array([]))

            if not peak_infos or mean_spec.size == 0:
                continue

            with st.expander(
                f"**{sel_serotype}** — Mean spectrum & diagnostics", expanded=True
            ):
                # Anchor & Window Plot
                fig_anchor, ax_anchor = plt.subplots(figsize=(14, 4))
                ax_anchor.plot(
                    raman_x,
                    mean_spec,
                    color="C0",
                    linewidth=1.5,
                    label=f"Mean ({sel_serotype}, high conc)",
                )
                for i, info in enumerate(peak_infos):
                    ax_anchor.axvline(
                        info.center,
                        color=f"C{(i % 9) + 1}",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.8,
                        label=f"{info.peak_name} @ {info.center:.0f} cm⁻¹",
                    )
                    ax_anchor.axvspan(
                        info.window_min,
                        info.window_max,
                        alpha=0.15,
                        color=f"C{(i % 9) + 1}",
                    )
                ax_anchor.set_xlabel("Raman shift (cm⁻¹)")
                ax_anchor.set_ylabel("Intensity")
                ax_anchor.set_title(
                    f"{sel_serotype} | Dashed = Voted Centers, Shaded = Search Windows"
                )
                ax_anchor.legend(loc="upper right", fontsize=8, ncol=2)
                ax_anchor.grid(True, alpha=0.3)
                fig_anchor.tight_layout()
                st.pyplot(fig_anchor, width="stretch")
                plt.close(fig_anchor)

                # Diagnostics Table
                diag_data = [
                    {
                        "Peak Name": p.peak_name,
                        "Center (cm⁻¹)": f"{p.center:.1f}",
                        "Window Range": f"[{p.window_min:.1f}, {p.window_max:.1f}]",
                        "Detection Success Rate (%)": f"{p.success_rate * 100:.1f}",
                    }
                    for p in peak_infos
                ]
                st.dataframe(pd.DataFrame(diag_data), width="stretch", hide_index=True)

        # ---- 3. Signal-Level Verification ----
        st.markdown("#### Signal-level verification")
        st.caption(
            "Inspect a single spectrum: shaded regions = serotype-specific search "
            "windows; green ★ = where the algorithm detected a peak (passed prominence "
            "check). Pick a serotype below to filter signals — this ties verification "
            "to the plots above."
        )
        wide_filtered = (
            wide_df.loc[filtered_features.index]
            if not filtered_features.empty
            else wide_df
        )
        sensor_col = "sensor_id" if "sensor_id" in filtered_features.columns else None
        conc_col = (
            "concentration_group"
            if "concentration_group" in filtered_features.columns
            else None
        )
        serotype_col = "serotype" if "serotype" in filtered_features.columns else None

        if sensor_col and conc_col and not wide_filtered.empty:
            sero_opts_ver = sorted(peak_by_sero.keys())
            sel_sero_ver = st.selectbox(
                "Serotype (filter signals)",
                options=sero_opts_ver,
                index=0,
                key="peak_diag_serotype_filter",
                help="Only show sensors/concentrations that have signals of this serotype.",
            )
            df_ver = (
                filtered_features[
                    filtered_features[serotype_col].astype(str) == sel_sero_ver
                ]
                if serotype_col
                else filtered_features
            )

            sensor_opts = (
                sorted(df_ver[sensor_col].dropna().unique().astype(str).tolist())
                if not df_ver.empty
                else []
            )
            conc_opts = (
                order_concentration_labels(
                    df_ver[conc_col].dropna().unique().astype(str).tolist()
                )
                or ["(none)"]
                if not df_ver.empty
                else ["(none)"]
            )

            if not sensor_opts or not conc_opts:
                st.info(
                    f"No signals for serotype **{sel_sero_ver}** in the current filters. "
                    "Adjust filters or select another serotype."
                )
            else:
                col_s, col_c, col_sig = st.columns(3)
                with col_s:
                    sel_sensor = st.selectbox(
                        "Sensor ID",
                        options=sensor_opts,
                        index=0,
                        key="peak_diag_sensor",
                    )
                with col_c:
                    sel_conc = st.selectbox(
                        "Concentration",
                        options=conc_opts,
                        index=0,
                        key="peak_diag_conc",
                    )
                matches = df_ver[
                    (df_ver[sensor_col].astype(str) == sel_sensor)
                    & (df_ver[conc_col].astype(str) == sel_conc)
                ]
                signal_labels = []
                for idx in matches.index:
                    fn = (
                        wide_filtered.loc[idx, "filename"]
                        if "filename" in wide_filtered.columns
                        else str(idx)
                    )
                    signal_labels.append(f"{fn} (idx {idx})")

                with col_sig:
                    if len(signal_labels) > 1:
                        sel_idx = st.selectbox(
                            "Signal",
                            options=range(len(signal_labels)),
                            format_func=lambda i: signal_labels[i],
                            key="peak_diag_signal",
                        )
                        row_idx = matches.index[sel_idx]
                    else:
                        row_idx = matches.index[0] if len(matches) > 0 else None

                if row_idx is not None:
                    row_sero = (
                        str(wide_filtered.loc[row_idx, serotype_col])
                        if serotype_col and serotype_col in wide_filtered.columns
                        else sel_sero_ver
                    )
                    row_peak_infos = peak_by_sero.get(row_sero)
                    if not row_peak_infos:
                        row_peak_infos = peak_by_sero.get(
                            st.session_state.get("peak_default_serotype")
                        )
                    if not row_peak_infos:
                        row_peak_infos = next(iter(peak_by_sero.values()), [])

                    spec_row = wide_filtered.loc[[row_idx]]
                    sig_mat = get_signals_matrix(spec_row)
                    raman = get_raman_shift(spec_row)
                    y_spec = sig_mat[0]
                    x_spec = np.asarray(raman, dtype=float)

                    # #region agent log
                    # Use only finite (x,y) points: concatenated files produce union columns,
                    # so each row has NaN where its source file lacked that Raman shift.
                    # Matplotlib breaks the line at NaN; filter to valid points for continuity.
                    valid = np.isfinite(y_spec.astype(float))
                    x_plot = x_spec[valid]
                    y_plot = np.asarray(y_spec, dtype=float)[valid]
                    sort_idx = np.argsort(x_plot)
                    x_plot = x_plot[sort_idx]
                    y_plot = y_plot[sort_idx]

                    fig_sig, ax_sig = plt.subplots(figsize=(14, 5))
                    ax_sig.plot(
                        x_plot, y_plot, color="C0", linewidth=1.2, label="Raw spectrum"
                    )

                    peak_cols = get_peak_height_columns(row_peak_infos)
                    for i, info in enumerate(row_peak_infos):
                        ax_sig.axvspan(
                            info.window_min,
                            info.window_max,
                            alpha=0.12,
                            color=f"C{(i % 9) + 1}",
                        )
                        peak_col = info.peak_name
                        if peak_col in filtered_features.columns:
                            val = filtered_features.loc[row_idx, peak_col]
                            if pd.notna(val) and np.isfinite(val):
                                mask = (x_spec >= info.window_min) & (
                                    x_spec <= info.window_max
                                )
                                window_y = np.where(mask, y_spec.astype(float), np.nan)
                                if mask.any() and np.any(np.isfinite(window_y)):
                                    local_idx = int(np.nanargmax(window_y))
                                    peak_x = float(x_spec[local_idx])
                                    peak_y = float(y_spec[local_idx])
                                    ax_sig.scatter(
                                        peak_x,
                                        peak_y,
                                        marker="*",
                                        s=200,
                                        color="green",
                                        edgecolors="darkgreen",
                                        linewidths=1.5,
                                        zorder=5,
                                        label="Detected" if i == 0 else None,
                                    )

                    ax_sig.set_xlabel("Raman shift (cm⁻¹)")
                    ax_sig.set_ylabel("Intensity")
                    ax_sig.set_title(
                        f"Signal: {sel_sensor} @ {sel_conc} ({row_sero}) | Green ★ = detected peaks"
                    )
                    ax_sig.legend(loc="upper right", fontsize=8)
                    ax_sig.grid(True, alpha=0.3)
                    fig_sig.tight_layout()
                    st.pyplot(fig_sig, width="stretch")
                    plt.close(fig_sig)
                else:
                    st.warning(
                        "No matching signal for selected sensor and concentration."
                    )
        else:
            st.caption(
                "Signal-level verification requires sensor_id and concentration_group "
                "in the data."
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

        stats_feat_opts = [
            c for c in BASIC_FEATURE_COLUMNS if c in filtered_features.columns
        ]
        stats_peak_opts = [
            c
            for c in get_peak_height_columns(
                next(
                    iter(st.session_state.get("peak_infos_by_serotype", {}).values()),
                    [],
                )
            )
            if c in filtered_features.columns
        ]
        stats_feat_opts = order_features_by_preference(
            stats_feat_opts + stats_peak_opts
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            feature_col = st.selectbox(
                "Feature (Y-axis)",
                options=stats_feat_opts if stats_feat_opts else ["(no features)"],
                index=min(2, len(stats_feat_opts) - 1) if stats_feat_opts else 0,
                key="stats_feature",
            )
        with col2:
            x_col = st.selectbox(
                "X-axis",
                options=x_opts if x_opts else ["(no grouping)"],
                index=min(x_default_idx, len(x_opts) - 1) if x_opts else 0,
                key="stats_x",
            )
        with col3:
            hue_default_s = pick_preferred_column(x_opts, ("serotype",)) or "None"
            hue_col_s = st.selectbox(
                "Hue",
                options=hue_opts,
                index=hue_opts.index(hue_default_s),
                key="stats_hue",
            )
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
                st.pyplot(fig_stats, width="stretch")
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
    feat_cols_avail = feat_cols_avail + [
        c
        for c in get_peak_height_columns(st.session_state.get("peak_infos", []))
        if c in filtered_features.columns
    ]
    feat_cols_avail = order_features_by_preference(feat_cols_avail)
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
                    st.dataframe(consistency_tbl, width="stretch", hide_index=True)
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
                        st.dataframe(deg_tbl, width="stretch", hide_index=True)

                    fig_deg = plot_degradation_trend(
                        df_deg,
                        assess_feature,
                        "test_ordinal",
                        group_col="sensor_id"
                        if "sensor_id" in df_deg.columns
                        else None,
                    )
                    st.pyplot(fig_deg, width="stretch")
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

                    st.dataframe(batch_tbl, width="stretch", hide_index=True)
                    if not deviating.empty:
                        st.markdown("**Deviating sensors (|z| > 2)**")
                        st.dataframe(deviating, width="stretch", hide_index=True)

                    fig_batch = plot_batch_boxplot(
                        assessment_df,
                        batch_feature,
                        sensor_col="sensor_id",
                        group_col=None,
                    )
                    st.pyplot(fig_batch, width="stretch")
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

# ---------------------------------------------------------------------------
# 5. Model-Based Sensor Consistency
# ---------------------------------------------------------------------------
with tab_model_consistency:
    mc_feat_cols = [c for c in BASIC_FEATURE_COLUMNS if c in filtered_features.columns]
    peak_cols_mc = get_peak_height_columns(
        next(iter(st.session_state.get("peak_infos_by_serotype", {}).values()), [])
    )
    mc_feat_cols = order_features_by_preference(
        mc_feat_cols + [c for c in peak_cols_mc if c in filtered_features.columns]
    )
    has_sensor = "sensor_id" in filtered_features.columns
    has_serotype = "serotype" in filtered_features.columns
    has_log_conc = "log_concentration" in filtered_features.columns

    if not mc_feat_cols:
        st.warning(
            "No feature columns available. Load data with Raman intensity columns "
            "and ensure filters yield samples."
        )
    elif not has_sensor or not has_serotype:
        st.warning(
            "Model-Based Consistency requires **sensor_id** and **serotype** columns. "
            "Ensure data is loaded with metadata and preprocess_metadata has run."
        )
    elif not has_log_conc:
        st.warning(
            "Model-Based Consistency requires **log_concentration**. "
            "Ensure preprocess_metadata has run on the loaded data."
        )
    else:
        st.markdown(
            "#### Model-based sensor consistency\n"
            "Two-pass regression with residual-based outlier removal. 0 CFU samples "
            "are excluded from the fit and shown as a horizontal baseline. Outliers "
            "are identified via IQR on absolute residuals and excluded from the "
            "clean fit."
        )
        sensor_opts = sorted(
            filtered_features["sensor_id"].dropna().unique().astype(str).tolist()
        ) or ["(none)"]
        serotype_opts = sorted(
            filtered_features["serotype"].dropna().unique().astype(str).tolist()
        ) or ["(none)"]

        mc_sensor, mc_serotype, mc_feature = st.columns(3)
        with mc_sensor:
            model_sensor = st.selectbox(
                "Sensor ID",
                options=sensor_opts,
                index=0,
                key="model_consistency_sensor",
            )
        with mc_serotype:
            model_serotype = st.selectbox(
                "Serotype",
                options=serotype_opts,
                index=0,
                key="model_consistency_serotype",
            )
        with mc_feature:
            model_feature = st.selectbox(
                "Feature to assess",
                options=mc_feat_cols,
                index=0,
                key="model_consistency_feature",
            )

        _mc_sensor_ok = model_sensor and model_sensor != "(none)"
        _mc_serotype_ok = model_serotype and model_serotype != "(none)"
        if _mc_sensor_ok and _mc_serotype_ok:
            model_df = filtered_features[
                (filtered_features["sensor_id"].astype(str) == model_sensor)
                & (filtered_features["serotype"].astype(str) == model_serotype)
            ].copy()
        else:
            model_df = pd.DataFrame()

        if model_df.empty and (_mc_sensor_ok and _mc_serotype_ok):
            st.warning(
                f"No samples for sensor_id={model_sensor}, serotype={model_serotype}. "
                "Adjust filters or selection."
            )
        elif not _mc_sensor_ok or not _mc_serotype_ok:
            st.info(
                "Select a sensor ID and serotype above to run model-based consistency."
            )
        else:
            cres = fit_concentration_regression_cleaned(model_df, model_feature)
            zero_baseline = get_zero_cfu_baseline(model_df, model_feature)

            if cres is not None:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Raw RMSE", f"{cres.raw_rmse:.4f}")
                with m2:
                    st.metric("Raw R²", f"{cres.raw_r2:.4f}")
                with m3:
                    st.metric("Clean RMSE", f"{cres.clean_rmse:.4f}")
                with m4:
                    st.metric("Clean R²", f"{cres.clean_r2:.4f}")
                if cres.n_outliers > 0:
                    st.caption(
                        f"Outliers dropped: {cres.n_outliers} (IQR on |residuals|)"
                    )
            else:
                st.warning(
                    "Insufficient data for regression (need ≥2 samples with valid "
                    "log concentration). 0 CFU samples are excluded from the fit."
                )

            try:
                fig_mc = plot_concentration_regression(
                    model_df,
                    model_feature,
                    regression_result=cres.clean_result if cres else None,
                    raw_regression_result=cres.raw_result if cres else None,
                    zero_cfu_baseline=zero_baseline,
                    outlier_mask=cres.outlier_mask if cres else None,
                    title=f"{model_sensor} — {model_serotype}",
                )
                st.pyplot(fig_mc, width="stretch")
            except ValueError as e:
                st.error(f"Plot error: {e}")

        # ---- Global Multi-Sensor Assessment ----
        st.markdown("---")
        st.markdown("#### Global Multi-Sensor Assessment")
        st.caption(
            "Per-sensor QA with dual threshold. Excluded if: Clean RMSE > 2× batch "
            "median OR Clean R² < 0.80 (dead/flat sensor)."
        )

        global_qa_default = [
            f for f in DEFAULT_GLOBAL_QA_FEATURES if f in mc_feat_cols
        ] or (mc_feat_cols[:5] if mc_feat_cols else [])
        global_qa_selected = st.multiselect(
            "Features for Global QA Table (and PDF)",
            options=mc_feat_cols,
            default=global_qa_default,
            key="global_qa_features",
        )
        if not global_qa_selected:
            st.info("Select at least one feature to populate the Global QA Table.")
        global_qa_tbl, excluded_map = get_global_model_consistency_qa(
            filtered_features,
            feature_cols=global_qa_selected,
        )
        if not global_qa_tbl.empty:
            st.dataframe(
                global_qa_tbl,
                width="stretch",
                hide_index=True,
                column_config={
                    "outliers": st.column_config.NumberColumn("Outliers"),
                    "raw_rmse": st.column_config.NumberColumn(
                        "Raw RMSE", format="%.4f"
                    ),
                    "raw_r2": st.column_config.NumberColumn("Raw R²", format="%.4f"),
                    "clean_rmse": st.column_config.NumberColumn(
                        "Clean RMSE", format="%.4f"
                    ),
                    "clean_r2": st.column_config.NumberColumn(
                        "Clean R²", format="%.4f"
                    ),
                },
            )
        else:
            st.info(
                "No regression results. Ensure filtered data has ≥2 valid "
                "(>0 CFU) points per sensor × serotype × feature."
            )

        st.markdown("##### Multi-sensor regression overlay")
        st.caption(
            "Select serotypes and features. Excluded sensors (dashed gray); passing "
            "sensors (solid colors). Tight bundle = uniform batch."
        )
        overlay_sero_opts = [s for s in serotype_opts if s and s != "(none)"]
        overlay_feat_default = (
            ["integral_area"] if "integral_area" in mc_feat_cols else mc_feat_cols[:1]
        )
        overlay_sero, overlay_feat = st.columns(2)
        with overlay_sero:
            overlay_serotypes = st.multiselect(
                "Serotype _(overlay)_",
                options=overlay_sero_opts,
                default=overlay_sero_opts,
                key="overlay_serotype",
            )
        with overlay_feat:
            overlay_features = st.multiselect(
                "Feature _(overlay)_",
                options=mc_feat_cols,
                default=overlay_feat_default,
                key="overlay_feature",
            )

        overlay_items: list = []
        macro_items: list = []
        for sero in overlay_serotypes:
            for feat in overlay_features:
                excluded = excluded_map.get((str(sero), str(feat)), set())
                all_sens = set(
                    str(s)
                    for s in filtered_features[
                        filtered_features["serotype"].astype(str) == sero
                    ]["sensor_id"]
                    .dropna()
                    .unique()
                )
                pass_sens = all_sens - excluded

                st.markdown(f"**{sero} — {feat}**")
                try:
                    fig_ov = plot_multi_sensor_regression(
                        filtered_features,
                        sero,
                        feat,
                        excluded_sensors=excluded,
                    )
                    st.pyplot(fig_ov, width="stretch")
                    overlay_items.append(
                        {"fig": fig_ov, "serotype": sero, "feature": feat}
                    )
                except ValueError as e:
                    st.error(f"Overlay ({sero}, {feat}): {e}")

                st.markdown("**Macro batch regression**")
                try:
                    fig_macro, macro_res = plot_macro_batch_regression(
                        filtered_features,
                        sero,
                        feat,
                        pass_sens,
                    )
                    st.pyplot(fig_macro, width="stretch")
                    macro_items.append(
                        {
                            "fig": fig_macro,
                            "macro_result": macro_res,
                            "serotype": sero,
                            "feature": feat,
                        }
                    )
                    if macro_res is not None:
                        ma1, ma2, ma3, ma4, ma5 = st.columns(5)
                        with ma1:
                            st.metric(
                                "Raw Batch RMSE",
                                f"{macro_res.raw_batch_rmse:.4f}",
                            )
                        with ma2:
                            st.metric(
                                "Raw Batch R²",
                                f"{macro_res.raw_batch_r2:.4f}",
                            )
                        with ma3:
                            st.metric(
                                "Clean Batch RMSE",
                                f"{macro_res.clean_batch_rmse:.4f}",
                            )
                        with ma4:
                            st.metric(
                                "Clean Batch R²",
                                f"{macro_res.clean_batch_r2:.4f}",
                            )
                        with ma5:
                            st.metric(
                                "Macro Outliers",
                                f"{macro_res.n_macro_outliers}",
                            )
                except ValueError as e:
                    st.error(f"Macro ({sero}, {feat}): {e}")
                st.markdown("---")

        if not overlay_serotypes or not overlay_features:
            st.info("Select at least one serotype and one feature to generate plots.")

        # ---- Phase 1 QA PDF Report ----
        st.markdown("---")
        st.markdown("#### Phase 1 QA Report")
        if st.button("Generate Phase 1 QA Report", key="phase1_qa_pdf_btn"):
            try:
                pdf_bytes = build_phase1_qa_pdf(
                    global_qa_table=global_qa_tbl if not global_qa_tbl.empty else None,
                    overlay_items=overlay_items,
                    macro_items=macro_items,
                    report_title="Sensor Consistency & Quality Assurance Report",
                )
                st.session_state["phase1_qa_pdf"] = pdf_bytes
                st.success("Phase 1 QA report generated. Click Download below.")
            except Exception as e:
                st.error(f"Phase 1 QA report generation failed: {e}")

        if "phase1_qa_pdf" in st.session_state:
            st.download_button(
                label="Download Phase 1 QA PDF",
                data=st.session_state["phase1_qa_pdf"],
                file_name="phase1_qa_report.pdf",
                mime="application/pdf",
                key="phase1_qa_pdf_download",
            )
