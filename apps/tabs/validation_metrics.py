"""
Validation Metrics tab — SENS-D Metrics-docx Tables 1–3.
"""

from __future__ import annotations

import logging

import streamlit as st

from cache import build_cached_validation_artifacts
from components.shared_ui import render_dataframe_stretch
from sensd_sers_analysis.config import (
    CLASSIFICATION_INLIER_FEATURE,
    VALIDATION_ACCURACY_MIN_THRESHOLD,
    VALIDATION_MIN_EVAL_ROWS,
    VALIDATION_N_SPLITS,
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
    Render the Validation Metrics tab with Metrics-docx Tables 1–3.

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
        "Three sponsor tables matching **Metrics tables.docx**:\n"
        "1. Concentration & repeatability (identification Accuracy)\n"
        "2. Quantification (CFU ranges, FP/FN, Meet target?)\n"
        "3. Consistency & reusability (reuse over time)"
    )
    with st.expander("Terms (same glossary as Sensor QC / Assessment)", expanded=False):
        st.markdown(
            "- **Repeatability CV%** — within-sensor signal precision "
            "*(protocol Exp. 1)*.\n"
            "- **Quantification** — predicted CFU range + FP/FN "
            "*(docx Table 2)*.\n"
            "- **Reuse / reusability** — Table 3 first→last change, failed "
            "sensors *(protocol Exp. 3)*.\n"
            "- **Accuracy** — mean over repeated **sensor-holdout** rounds."
        )
    st.info(
        f"ML metrics use **{VALIDATION_N_SPLITS}× sensor-holdout** "
        f"(~20% sensors each round); reported rates are the **mean across "
        f"rounds**, blanked when total held-out **Eval N** < "
        f"{VALIDATION_MIN_EVAL_ROWS}. CV / counts use all clean Pass+inlier rows."
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

    try:
        artifacts = build_cached_validation_artifacts(
            filtered_features,
            tuple(feat_cols),
            repeatability_feature=repeatability_feature,
        )
    except ValueError as exc:
        logger.exception("Validation table generation failed")
        st.error(f"Validation unavailable: {exc}")
        return

    preds = artifacts.predictions
    if preds and preds.sensor_holdout_available:
        holdout_caption = (
            f"ML: **{preds.n_splits}** sensor-holdout rounds; "
            f"~**{preds.n_test_sensors:.1f}** test sensors / round; "
            f"**{preds.n_eval_rows}** total held-out eval rows (summed)."
        )
    else:
        holdout_caption = "ML sensor holdout unavailable (need ≥2 sensors)."

    st.caption(
        f"Clean rows: **{artifacts.n_classification_rows}** classification, "
        f"**{artifacts.n_regression_rows}** regression (positive CFU). "
        f"Pass/Fail threshold: **{VALIDATION_ACCURACY_MIN_THRESHOLD * 100:.0f}%** "
        f"identification accuracy. {holdout_caption}"
    )

    if artifacts.n_classification_rows == 0:
        st.warning(
            "No clean rows for validation. Ensure sensors pass QA and both "
            "positive-CFU serotypes and Rinsate (0 CFU) samples are present."
        )
        return

    st.markdown("---")
    st.markdown("##### Table 1 — Concentration and Repeatability Testing *(docx / Exp. 1)*")
    st.caption(
        "Grouped by serovar × **target** CFU. Repeatability CV% = signal precision; "
        "Concentration CV% = sample spread (our addition). Accuracy = mean over "
        "sensor-holdout rounds."
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
    st.markdown("##### Table 2 — Quantification *(docx)*")
    st.caption(
        "Same serovar × target rows. Quantification Accuracy = pooled predicted "
        "CFU range (~lo-hi) on held-out positives; FP/FN and Meet Target? use "
        "mean identification rates across sensor-holdout rounds."
    )
    if artifacts.quantification.empty:
        st.info("No quantification rows for the current filters.")
    else:
        render_dataframe_stretch(artifacts.quantification)
        _csv_download_button(
            artifacts.quantification,
            "Download Table 2 (CSV)",
            "validation_table2_quantification.csv",
            "validation_table2_csv",
        )

    st.markdown("---")
    st.markdown("##### Table 3 — Consistency & Sensor Reusability *(docx / Exp. 3)*")
    st.caption(
        "Sensors with **≥2** distinct `test_id` at the same serovar × target "
        "CFU. First→last signal change = reuse drift; Failed = Excluded in "
        "between-sensor QA."
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
            "Download Table 3 (CSV)",
            "validation_table3_consistency_reusability.csv",
            "validation_table3_csv",
        )
