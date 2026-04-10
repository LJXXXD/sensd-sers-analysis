"""
Feature Analysis tab — distributions, inter-feature correlations, and concentration links.
"""

import logging

import streamlit as st

from components.shared_ui import render_dataframe_stretch, render_figure_stretch
from sensd_sers_analysis.processing import (
    BASIC_FEATURE_COLUMNS,
    get_available_feature_columns,
    get_feature_metadata_columns,
    pick_preferred_column,
)
from sensd_sers_analysis.visualization import (
    compute_feature_log_concentration_correlations,
    plot_feature_correlation_heatmap,
    plot_feature_distribution,
    plot_feature_log_concentration_correlation_bars,
    plot_feature_log_concentration_scatter,
)
from theme import (
    DEFAULT_FIGSIZE_WIDTH,
    PLOT_HEIGHT_DEFAULT,
    PLOT_HEIGHT_MAX,
    PLOT_HEIGHT_MIN,
)

logger = logging.getLogger(__name__)


def render(filtered_features, peak_artifacts):
    """
    Render the Feature Analysis tab.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe for the current app state.
    peak_artifacts:
        Shared peak artifacts from the derived data bundle.
    """
    st.markdown(
        "Summarize extracted features: **distributions** by group, **pairwise correlations** "
        "among numeric features, and **Pearson association** with **log concentration** when "
        "that column is available."
    )

    all_feat_nan = all(
        filtered_features[c].isna().all()
        for c in BASIC_FEATURE_COLUMNS
        if c in filtered_features.columns
    )
    if all_feat_nan:
        logger.warning("All extracted features are NaN")
        st.warning(
            "All extracted features are NaN. This usually means the loaded data "
            "lacks Raman intensity columns (rs_*) or they contain no valid "
            "numeric values. Ensure your Excel files use the expected embedded "
            "format with Raman shift columns."
        )
        return

    stats_feat_opts = get_available_feature_columns(
        filtered_features,
        peak_artifacts.peak_infos_by_serotype,
    )

    st.subheader("Feature distributions")
    st.caption("Box or violin plots by metadata grouping.")
    st.markdown("##### Plot options")
    x_opts = get_feature_metadata_columns(filtered_features)
    hue_opts = ["None"] + x_opts

    x_default = pick_preferred_column(x_opts) or (x_opts[0] if x_opts else None)
    x_default_idx = x_opts.index(x_default) if x_default in x_opts else 0

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
    col_plot_type, col_height = st.columns(2)
    with col_plot_type:
        plot_type = st.radio(
            "Plot type",
            options=["box", "violin"],
            index=0,
            horizontal=True,
            key="stats_plot_type",
        )
    with col_height:
        plot_height = st.slider(
            "Height (in)",
            min_value=PLOT_HEIGHT_MIN,
            max_value=PLOT_HEIGHT_MAX,
            value=PLOT_HEIGHT_DEFAULT,
            step=1,
            key="stats_plot_height",
        )

    if filtered_features.empty:
        st.warning("No samples match the selected filters for feature analysis.")
        return

    x_val = (
        None
        if (not x_col or x_col == "(no grouping)" or x_col not in filtered_features.columns)
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
            figsize=(DEFAULT_FIGSIZE_WIDTH, plot_height),
        )
        render_figure_stretch(fig_stats)
    except ValueError as e:
        logger.warning("Feature distribution plot error: %s", e)
        st.error(f"Plot error: {e}")

    st.divider()
    st.subheader("Correlation among features")
    st.caption("Pearson **r** on pairwise-complete rows (heatmap shows the lower triangle).")
    default_corr = stats_feat_opts[: min(12, len(stats_feat_opts))]
    selected_corr = st.multiselect(
        "Features for correlation matrix",
        options=stats_feat_opts,
        default=default_corr,
        key="fa_corr_features",
    )
    if len(selected_corr) >= 2:
        heat_h = max(5.0, 0.55 * len(selected_corr))
        try:
            fig_corr = plot_feature_correlation_heatmap(
                filtered_features,
                selected_corr,
                figsize=(DEFAULT_FIGSIZE_WIDTH, heat_h),
            )
            render_figure_stretch(fig_corr)
        except ValueError as e:
            logger.warning("Feature correlation heatmap error: %s", e)
            st.warning(str(e))
    else:
        st.caption("Select at least two features to draw the matrix.")

    st.divider()
    st.subheader("Association with log concentration")
    if "log_concentration" not in filtered_features.columns:
        st.info(
            "No **log_concentration** column in the filtered table. After metadata preprocessing, "
            "this column appears for rows with positive CFU/ml; relax filters if you expect "
            "concentration-labeled samples."
        )
    elif not stats_feat_opts:
        st.warning("No feature columns available for concentration association.")
    else:
        st.caption(
            "Bar chart: Pearson **r** vs **log_concentration** per feature (≥3 paired finite "
            "points, non-degenerate variance). Scatter: one feature with a single OLS line on "
            "all points (exploratory, not outlier-robust)."
        )
        corr_tab = compute_feature_log_concentration_correlations(
            filtered_features,
            stats_feat_opts,
        )
        if corr_tab.empty:
            st.warning(
                "Could not compute feature–log concentration correlations "
                "(need ≥3 valid paired points per feature)."
            )
        else:
            try:
                fig_bars = plot_feature_log_concentration_correlation_bars(
                    filtered_features,
                    stats_feat_opts,
                )
                render_figure_stretch(fig_bars)
            except ValueError as e:
                logger.warning("Correlation bar plot error: %s", e)
                st.warning(str(e))
            render_dataframe_stretch(
                corr_tab,
                column_config={
                    "pearson_r": st.column_config.NumberColumn("Pearson r", format="%.4f"),
                    "p_value": st.column_config.NumberColumn("p-value", format="%.4g"),
                    "n": st.column_config.NumberColumn("n", format="%d"),
                },
            )

        st.markdown("##### Scatter with OLS")
        sc_col_a, sc_col_b, sc_col_c = st.columns(3)
        with sc_col_a:
            sc_feature = st.selectbox(
                "Feature (y-axis)",
                options=stats_feat_opts,
                index=0,
                key="fa_scatter_feature",
            )
        hue_scatter_opts = ["None"] + [
            c for c in ("serotype", "sensor_id") if c in filtered_features.columns
        ]
        with sc_col_b:
            sc_hue = st.selectbox(
                "Hue",
                options=hue_scatter_opts,
                index=0,
                key="fa_scatter_hue",
            )
        with sc_col_c:
            sc_height = st.slider(
                "Scatter height (in)",
                min_value=PLOT_HEIGHT_MIN,
                max_value=PLOT_HEIGHT_MAX,
                value=min(7, PLOT_HEIGHT_DEFAULT),
                step=1,
                key="fa_scatter_height",
            )
        hue_sc = None if sc_hue == "None" else sc_hue
        try:
            fig_sc = plot_feature_log_concentration_scatter(
                filtered_features,
                sc_feature,
                hue_col=hue_sc,
                figsize=(DEFAULT_FIGSIZE_WIDTH, sc_height),
            )
            render_figure_stretch(fig_sc)
        except ValueError as e:
            logger.warning("Feature vs log concentration scatter error: %s", e)
            st.warning(str(e))
