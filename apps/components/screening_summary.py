"""Visual explanation of sensor screening and its underlying measurements."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cache import build_cached_global_qa_artifacts, build_cached_single_sensor_consistency_artifacts
from sensd_sers_analysis.application.contracts import ModelConsistencySelection
from sensd_sers_analysis.application.sensor_assessment_service import (
    build_screening_counts,
    build_screening_point_ledger,
)
from sensd_sers_analysis.config.model_policies import (
    CLASSIFICATION_INLIER_FEATURE,
    GLOBAL_QA_R2_MIN_THRESHOLD,
    GLOBAL_QA_REJECTION_MULTIPLIER,
)
from sensd_sers_analysis.visualization import plot_concentration_regression, plot_spectra
from sensd_sers_analysis.visualization.inventory_plots import plot_screening_counts


def render_screening_summary(features: pd.DataFrame, tidy: pd.DataFrame) -> None:
    """Show current-filter QA outcomes and drill into actual sensor measurements."""
    feature = CLASSIFICATION_INLIER_FEATURE
    if features.empty or feature not in features:
        st.info("No spectra available for screening.")
        return
    qa = build_cached_global_qa_artifacts(features, (feature,)).table
    counts = build_screening_counts(features, qa)
    st.subheader("Which sensors pass screening?")
    totals = counts.groupby("status")["spectra"].sum()
    for column, label, value in zip(
        st.columns(4),
        ("Sensors", "Passing spectra", "Excluded spectra", "Not assessed"),
        (
            features.sensor_id.nunique(),
            totals.get("Pass", 0),
            totals.get("Excluded", 0),
            totals.get("Not assessed", 0),
        ),
    ):
        column.metric(label, int(value))
    st.caption("Current filters · Before final spectrum cleaning · No duplicate removal")
    figure = plot_screening_counts(counts)
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption(
        "Screening is per sensor and serotype; one sensor can have both passing and excluded measurements."
    )
    st.subheader("Why was this sensor excluded?")
    pairs = features[["sensor_id", "serotype"]].drop_duplicates().dropna()
    choices = list(pairs.itertuples(index=False, name=None))
    if not choices:
        return
    statuses = {
        (row.sensor_id, row.serotype): row.status
        if np.isfinite([row.clean_r2, row.clean_rmse]).all()
        else "Not assessed"
        for row in qa.itertuples()
    }
    choices.sort(key=lambda pair: (statuses.get(pair) != "Excluded", str(pair)))
    selected = st.selectbox(
        "Inspect sensor / serotype",
        choices,
        format_func=lambda pair: f"{pair[0]} · {pair[1]} · {statuses.get(pair, 'Not assessed')}",
        key="screening_inspect",
    )
    sensor, serotype = selected
    selected_qa = (
        qa.loc[(qa.sensor_id == sensor) & (qa.serotype == serotype)] if not qa.empty else qa
    )
    if not selected_qa.empty and statuses.get(selected) != "Not assessed":
        row = selected_qa.iloc[0]
        limit = (
            GLOBAL_QA_REJECTION_MULTIPLIER * qa.loc[qa.serotype == serotype, "clean_rmse"].median()
        )
        reasons = []
        if row.clean_r2 < GLOBAL_QA_R2_MIN_THRESHOLD:
            reasons.append(
                f"Response does not follow concentration consistently (R² {row.clean_r2:.2f} < {GLOBAL_QA_R2_MIN_THRESHOLD:.2f})."
            )
        if row.clean_rmse > limit:
            reasons.append(
                f"Scatter around the fitted trend is too large (RMSE {row.clean_rmse:.3g} > {limit:.3g})."
            )
        st.write(" ".join(reasons) or "No exclusion threshold was exceeded for this pair.")
    else:
        st.info("Not enough concentration variation to assess this sensor/serotype pair.")
    comparison_options = [None] + [
        pair
        for pair in choices
        if pair != selected and pair[1] == serotype and statuses.get(pair) == "Pass"
    ]
    comparison = st.selectbox(
        "Compare with a passing sensor",
        comparison_options,
        format_func=lambda pair: "None" if pair is None else str(pair[0]),
        key="screening_compare",
    )
    shown = [selected] + ([comparison] if comparison else [])
    figure, axes = plt.subplots(
        1, len(shown), figsize=(12, 5), squeeze=False, sharex=True, sharey=True
    )
    artifacts = None
    for axis, pair in zip(axes.flat, shown):
        item = build_cached_single_sensor_consistency_artifacts(
            features, ModelConsistencySelection(pair[0], pair[1], feature)
        )
        if pair == selected:
            artifacts = item
        result = item.regression_result
        try:
            plot_concentration_regression(
                item.model_df,
                feature,
                regression_result=result.clean_result if result else None,
                raw_regression_result=result.raw_result if result else None,
                outlier_mask=result.outlier_mask if result else None,
                title=f"{pair[0]} · {pair[1]} · {statuses.get(pair, 'Not assessed')}",
                ax=axis,
            )
            axis.set_ylabel("Integrated spectral intensity (a.u. × cm⁻¹)")
        except ValueError:
            axis.set_title(f"{pair[0]}: no valid fit points")
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption(
        "Each dot is a measurement. The line shows its concentration trend; red × marks a point omitted from the fit. This is separate from excluding the entire sensor/serotype pair."
    )
    if artifacts is not None:
        ledger = build_screening_point_ledger(artifacts)
        with st.expander("Which individual measurements were used in this fit?"):
            columns = [
                c
                for c in (
                    "filename",
                    "signal_index",
                    "source_filename",
                    "concentration",
                    feature,
                    "fit_use",
                )
                if c in ledger
            ]
            st.dataframe(ledger[columns], hide_index=True, width="stretch")
    if not tidy.empty:
        spectra = tidy.loc[(tidy.sensor_id == sensor) & (tidy.serotype == serotype)]
        st.markdown(f"#### Spectra from {sensor} · {serotype}")
        if not spectra.empty:
            figure = plot_spectra(
                spectra, hue="target_concentration_group", show_variance=False, figsize=(12, 5)
            )
            st.pyplot(figure, width="stretch")
            plt.close(figure)
    with st.expander("Screening criteria and exact results"):
        if not tidy.empty:
            st.caption(
                f"Raman range: {tidy.raman_shift.min():g}–{tidy.raman_shift.max():g} cm⁻¹. QA reflects the current filters and feature range."
            )
        st.write(
            f"Feature: {feature}. Exclude if cleaned R² < {GLOBAL_QA_R2_MIN_THRESHOLD:.2f} or cleaned RMSE > {GLOBAL_QA_REJECTION_MULTIPLIER:g} × the median for the same serotype. Exclusion does not establish physical sensor failure."
        )
        st.dataframe(qa, hide_index=True, width="stretch")
