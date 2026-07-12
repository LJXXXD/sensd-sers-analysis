"""
Validation Metrics tab — SENS-D concentration and sensor-reusability tables.
"""

from __future__ import annotations

import logging

import streamlit as st

from cache import build_cached_validation_artifacts
from components.shared_ui import render_dataframe_stretch
from sensd_sers_analysis.config import (
    CLASSIFICATION_INLIER_FEATURE,
    VALIDATION_ACCURACY_MIN_THRESHOLD,
)
from sensd_sers_analysis.processing import (
    CLASSIFICATION_FEATURE_BASE,
    list_targeted_peak_feature_columns,
)

logger = logging.getLogger(__name__)


def _csv_download_button(df, label: str, filename: str, key: str) -> None:
    """Render a CSV download button when the dataframe is non-empty."""
    if df.empty:
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def render(filtered_features, peak_artifacts) -> None:
    """
    Render the Validation Metrics tab with Tables 1 and 2.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe for the current app state.
    peak_artifacts:
        Shared peak artifacts (used to resolve available feature columns).
    """
    _ = peak_artifacts

    st.markdown(
        "#### Validation Metrics\n"
        "Summary tables for **concentration/repeatability** (Table 1) and "
        "**sensor reusability** (Table 2). Uses Pass-sensor, inlier-cleaned rows "
        "and the same feature set as serotype classification."
    )

    has_required = (
        "sensor_id" in filtered_features.columns
        and "serotype" in filtered_features.columns
        and "concentration_group" in filtered_features.columns
        and "test_id" in filtered_features.columns
        and "PC1" in filtered_features.columns
    )
    peak_cols = list_targeted_peak_feature_columns(filtered_features.columns)
    feat_cols = [
        c for c in CLASSIFICATION_FEATURE_BASE + peak_cols if c in filtered_features.columns
    ]

    if not has_required:
        st.warning(
            "Validation metrics require **sensor_id**, **serotype**, "
            "**concentration_group**, **test_id**, and **PC1**. "
            "Load embedded Excel data and ensure preprocessing has run."
        )
        return
    if len(feat_cols) < 2:
        st.warning("Need at least 2 feature columns for validation models.")
        return

    repeatability_feature = st.selectbox(
        "Repeatability feature",
        options=[c for c in feat_cols if c in filtered_features.columns],
        index=(
            feat_cols.index(CLASSIFICATION_INLIER_FEATURE)
            if CLASSIFICATION_INLIER_FEATURE in feat_cols
            else 0
        ),
        key="validation_repeatability_feature",
        help="Scalar feature for CV% and first-to-last signal change in the tables.",
    )

    artifacts = build_cached_validation_artifacts(
        filtered_features,
        tuple(feat_cols),
        repeatability_feature=repeatability_feature,
    )

    st.caption(
        f"Clean rows: **{artifacts.n_classification_rows}** classification, "
        f"**{artifacts.n_regression_rows}** regression (positive CFU). "
        f"Pass/Fail threshold: **{VALIDATION_ACCURACY_MIN_THRESHOLD * 100:.0f}%** accuracy. "
        + (
            f"ML metrics evaluated on **{artifacts.predictions.n_eval_rows}** held-out "
            f"test-sensor rows ({artifacts.predictions.n_test_sensors} sensors)."
            if artifacts.predictions and artifacts.predictions.sensor_holdout_available
            else "ML metrics use all rows (sensor holdout unavailable)."
        )
    )

    if artifacts.n_classification_rows == 0:
        st.warning(
            "No clean rows for validation. Ensure sensors pass QA and both "
            "positive-CFU serotypes and Rinsate (0 CFU) samples are present."
        )
        return

    st.markdown("---")
    st.markdown("##### Table 1 — Concentration and Repeatability Testing")
    st.caption(
        "Grouped by serovar and **target** concentration (binned 0, 1, 10, 100, 1000 CFU/mL). "
        "Each serovar block ends with an **Overall** row. One global classifier and regressor "
        "are trained on held-in sensors, then applied to the full dataset; accuracy, FP/FN, "
        "and quantification ranges are computed on held-out test sensors only."
    )
    if artifacts.concentration_repeatability.empty:
        st.info("No concentration/repeatability rows for the current filters.")
    else:
        render_dataframe_stretch(artifacts.concentration_repeatability)
        _csv_download_button(
            artifacts.concentration_repeatability,
            "Download Table 1 (CSV)",
            "validation_table1_concentration_repeatability.csv",
            "validation_table1_csv",
        )

    st.markdown("---")
    st.markdown("##### Table 2 — Consistency Testing & Sensor Reusability")
    st.caption(
        "Sensors with **≥2 repeated tests** (distinct test_id) at the same "
        "serovar × target concentration (0, 1, 10, 100, 1000 CFU/mL). "
        "Failed sensors are those Excluded in global QA."
    )
    if artifacts.consistency_reusability.empty:
        st.info(
            "No reusability rows — need sensors with at least two repeated "
            "test_id values at the same serovar and concentration."
        )
    else:
        render_dataframe_stretch(artifacts.consistency_reusability)
        _csv_download_button(
            artifacts.consistency_reusability,
            "Download Table 2 (CSV)",
            "validation_table2_consistency_reusability.csv",
            "validation_table2_csv",
        )
