"""
Sensor QC tab — per-group consistency (signal CV vs. concentration CV),
degradation, and multi-sensor batch stability.

Replicates are grouped by nominal **target** concentration so sample-to-sample
CFU spread stays inside each group. The consistency table reports the SERS
signal CV alongside the actual-concentration CV, exposing how much variability
the sensor adds beyond the inherent sample variability.
"""

import logging
from dataclasses import replace

import streamlit as st

from cache import build_cached_sensor_assessment_artifacts
from components.shared_ui import (
    render_dataframe_stretch,
    render_figure_stretch,
    render_pdf_download_section,
)

from sensd_sers_analysis.application import (
    SensorAssessmentSelection,
    build_sensor_assessment_pdf_bytes,
)
from sensd_sers_analysis.config import BATCH_DEVIATION_Z_THRESHOLD
from sensd_sers_analysis.utils import order_concentration_labels
from sensd_sers_analysis.processing import get_available_feature_columns
from sensd_sers_analysis.visualization import (
    plot_degradation_trend,
    plot_sensor_batch_stability,
    plot_signal_vs_concentration_cv,
)

logger = logging.getLogger(__name__)

# Grouping columns and their friendly labels for the consistency table.
_GROUP_LABELS: dict[str, str] = {
    "sensor_id": "Sensor",
    "serotype": "Serotype",
    "target_concentration_group": "Target CFU",
}

# Degradation defaults: intensity-based features track carryover/loss best, and
# low concentrations expose contamination (rising baseline) most clearly.
_DEGRADATION_PREFERRED_FEATURES = ("max_intensity", "mean_intensity")
_DEGRADATION_PREFERRED_CONCENTRATIONS = ("0 CFU", "1 CFU")


def _default_index(options: list[str], preferred: tuple[str, ...]) -> int:
    """Return the index of the first preferred option present, else 0."""
    for candidate in preferred:
        if candidate in options:
            return options.index(candidate)
    return 0


def _format_consistency_table(table):
    """
    Pivot the long per-feature consistency result into one row per group.

    Each selected feature contributes a ``<feature> signal CV%`` column
    (outlier-filtered) placed side by side, and the feature-independent
    actual-concentration CV% is appended once at the end. This lets a single row
    compare signal spread across feature representations against the underlying
    sample (CFU) spread.
    """
    if table.empty:
        return table

    group_cols = [c for c in _GROUP_LABELS if c in table.columns]
    if not group_cols or "feature" not in table.columns or "cv_filtered" not in table.columns:
        return table

    # Feature-independent scalars are identical across a group's feature rows.
    base_value_cols = [c for c in ("n_total", "conc_mean", "conc_cv_raw") if c in table.columns]
    base = (
        table[group_cols + base_value_cols]
        .drop_duplicates(subset=group_cols)
        .reset_index(drop=True)
    )

    # One signal-CV column per feature (outlier-filtered fraction -> percent).
    cv_wide = table.pivot_table(
        index=group_cols,
        columns="feature",
        values="cv_filtered",
        aggfunc="first",
    ).reset_index()
    feature_cols = [c for c in cv_wide.columns if c not in group_cols]

    merged = base.merge(cv_wide, on=group_cols, how="left")
    for col in feature_cols:
        merged[col] = (merged[col].astype(float) * 100).round(1)
    if "conc_cv_raw" in merged.columns:
        merged["conc_cv_raw"] = (merged["conc_cv_raw"].astype(float) * 100).round(1)

    ordered = (
        group_cols
        + [c for c in ("n_total",) if c in merged.columns]
        + feature_cols
        + [c for c in ("conc_mean", "conc_cv_raw") if c in merged.columns]
    )
    rename = dict(_GROUP_LABELS)
    rename["n_total"] = "N"
    rename["conc_mean"] = "Actual CFU mean"
    rename["conc_cv_raw"] = "Concentration CV%"
    rename.update({c: f"{c} signal CV%" for c in feature_cols})
    return merged[ordered].rename(columns=rename)


def render(filtered_features, peak_artifacts):
    """
    Render the Sensor QC tab.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe for the current app state.
    peak_artifacts:
        Shared peak artifacts from the derived data bundle.
    """

    feat_cols_avail = get_available_feature_columns(
        filtered_features,
        peak_artifacts.peak_infos_by_serotype,
    )
    has_serotype = "serotype" in filtered_features.columns
    has_target_group = "target_concentration_group" in filtered_features.columns

    if not feat_cols_avail:
        st.warning(
            "No feature columns available. Load data with Raman intensity columns "
            "and ensure filters yield samples."
        )
        return

    st.markdown(
        "Interactive per-group QC at a fixed **target** concentration. "
        "**Signal CV%** = SERS feature spread; **Concentration CV%** = actual "
        "plate-count CFU spread. Compare them as diagnostics; their difference does not "
        "isolate sensor variance."
    )
    with st.expander("Terms (aligned with the fiber-optics SERS testing protocol)", expanded=False):
        st.markdown(
            "- **Within-sensor repeatability** — same sensor, same target CFU; "
            "how stable the signal is across replicates *(protocol: "
            "repeatability / CV%)*.\n"
            "- **Reuse over time** — same sensor across successive tests; "
            "slope/drift and carryover *(protocol: consistency testing & "
            "reusability / failure)*.\n"
            "- **Between-sensor agreement** — do different sensors read alike "
            "under the same conditions *(protocol: sensor-to-sensor / among "
            "sensors repeatability)*."
        )

    if not has_serotype or not has_target_group:
        st.warning(
            "Sensor QC requires **serotype** and **target_concentration_group** "
            "columns. Ensure data is loaded with metadata and preprocess_metadata "
            "has run."
        )
        return

    st.markdown(
        "#### Experimental variable control\n"
        "Select a **specific serotype** and **target concentration** before running "
        "QC. Statistics are computed only on replicates sharing these conditions."
    )
    serotype_opts = sorted(
        filtered_features["serotype"].dropna().unique().astype(str).tolist()
    ) or ["(none)"]
    conc_raw = (
        filtered_features["target_concentration_group"].dropna().astype(str).unique().tolist()
    )
    conc_opts = [c for c in conc_raw if c and c != "nan"]
    conc_opts = order_concentration_labels(conc_opts) if conc_opts else ["(none)"]

    a_sero, a_conc, a_outlier = st.columns(3)
    with a_sero:
        assess_serotype = st.selectbox(
            "Serotype _(required)_",
            options=serotype_opts,
            index=0,
            key="assess_serotype",
        )
    with a_conc:
        assess_concentration = st.selectbox(
            "Target concentration _(required)_",
            options=conc_opts,
            index=0,
            key="assess_concentration",
        )
    with a_outlier:
        outlier_method = st.radio(
            "Outlier method",
            options=["iqr", "zscore"],
            index=0,
            horizontal=True,
            key="assess_outlier",
        )

    assess_features = st.multiselect(
        "Features (signal CV columns, side by side)",
        options=feat_cols_avail,
        default=feat_cols_avail,
        key="assess_features",
        help=(
            "Each selected feature adds one signal-CV column to the consistency "
            "table. The first selected feature drives the degradation trend."
        ),
    )

    _sero_valid = assess_serotype and assess_serotype != "(none)"
    _conc_valid = assess_concentration and assess_concentration != "(none)"
    if not _sero_valid or not _conc_valid:
        st.info("Select a specific serotype and target concentration above to run QC.")
        return
    if not assess_features:
        st.info("Select at least one feature to compute signal CV.")
        return

    primary_feature = assess_features[0]

    preview_selection = SensorAssessmentSelection(
        serotype=assess_serotype,
        target_concentration_group=assess_concentration,
        feature=primary_feature,
        outlier_method=outlier_method,
        batch_feature=primary_feature,
        consistency_features=tuple(assess_features),
    )
    preview_artifacts = build_cached_sensor_assessment_artifacts(
        filtered_features,
        preview_selection,
    )

    if preview_artifacts.assessment_df.empty:
        st.warning(
            f"No samples for serotype={assess_serotype}, "
            f"target concentration={assess_concentration}. Adjust filters or selection."
        )
        return

    artifacts = preview_artifacts

    st.markdown("##### Within-sensor repeatability (Signal CV vs. Concentration CV)")
    st.caption(
        f"Within serotype={assess_serotype}, target={assess_concentration}. "
        "One row per sensor; each selected feature is a **signal CV%** column "
        "(outlier-filtered), side by side with **Concentration CV%** "
        "(sample spread). Signal ≫ Concentration ⇒ sensor-added variability."
    )
    if artifacts.consistency_error:
        logger.warning("Consistency error: %s", artifacts.consistency_error)
        st.error(f"Consistency error: {artifacts.consistency_error}")
    elif not artifacts.display_consistency_table.empty:
        render_dataframe_stretch(_format_consistency_table(artifacts.display_consistency_table))
        st.caption(
            "Signal CV% (per feature) vs the sample **Concentration CV%** "
            "(hatched). Signal bars well above the concentration bar = "
            "sensor-added variability; bars with no height = undefined (e.g. a "
            "single replicate)."
        )
        try:
            fig_cv = plot_signal_vs_concentration_cv(artifacts.display_consistency_table)
            render_figure_stretch(fig_cv)
        except ValueError as exc:
            logger.warning("Signal-vs-concentration CV plot error: %s", exc)

    st.markdown("##### Reuse over time (degradation / carryover)")
    st.caption(
        "Same sensor across successive tests *(protocol: consistency & "
        "reusability)*. Own serotype / target / feature selectors; shared "
        "outlier method applies. Prefer **0 or 1 CFU** to catch carryover "
        "(rising baseline). Negative slope = signal loss. Outliers removed "
        "only within a test session — across-time jumps are kept."
    )
    d_sero, d_conc, d_feat = st.columns(3)
    with d_sero:
        deg_serotype = st.selectbox(
            "Serotype (degradation)",
            options=serotype_opts,
            index=(serotype_opts.index(assess_serotype) if assess_serotype in serotype_opts else 0),
            key="deg_serotype",
        )
    with d_conc:
        deg_concentration = st.selectbox(
            "Target concentration (degradation)",
            options=conc_opts,
            index=_default_index(conc_opts, _DEGRADATION_PREFERRED_CONCENTRATIONS),
            key="deg_concentration",
        )
    with d_feat:
        deg_feature = st.selectbox(
            "Feature (degradation)",
            options=feat_cols_avail,
            index=_default_index(feat_cols_avail, _DEGRADATION_PREFERRED_FEATURES),
            key="deg_feature",
        )

    deg_selection = SensorAssessmentSelection(
        serotype=deg_serotype,
        target_concentration_group=deg_concentration,
        feature=deg_feature,
        outlier_method=outlier_method,
        batch_feature=deg_feature,
        consistency_features=(deg_feature,),
    )
    deg_artifacts = build_cached_sensor_assessment_artifacts(filtered_features, deg_selection)

    if deg_artifacts.degradation_error:
        logger.warning("Degradation error: %s", deg_artifacts.degradation_error)
        st.error(f"Degradation error: {deg_artifacts.degradation_error}")
    elif deg_artifacts.degradation_input_df.empty or len(deg_artifacts.degradation_input_df) < 2:
        st.info(
            f"Insufficient temporal data for {deg_serotype} / {deg_feature} at "
            f"{deg_concentration} (need test_id or date with ≥2 tests)."
        )
    else:
        if not deg_artifacts.degradation_table.empty:
            render_dataframe_stretch(deg_artifacts.degradation_table)
        fig_deg = plot_degradation_trend(
            deg_artifacts.degradation_input_df,
            deg_feature,
            "test_ordinal",
            group_col=(
                "sensor_id" if "sensor_id" in deg_artifacts.degradation_input_df.columns else None
            ),
        )
        render_figure_stretch(fig_deg)

    st.markdown("---")
    st.markdown("#### Between-sensor agreement (sensor-to-sensor repeatability)")
    st.caption(
        f"Within serotype={assess_serotype}, target={assess_concentration}. "
        "Do sensors agree with each other? *(protocol: among-sensors "
        "repeatability)*. Each sensor's mean (± SD when n≥2) vs the batch "
        "consensus band; dots = individual reads. Outside the band = "
        "deviating."
    )
    if "sensor_id" in preview_artifacts.assessment_df.columns:
        batch_feature = st.selectbox(
            "Feature (batch)",
            options=feat_cols_avail,
            index=(
                feat_cols_avail.index(primary_feature) if primary_feature in feat_cols_avail else 0
            ),
            key="batch_feature",
        )
        selection = SensorAssessmentSelection(
            serotype=assess_serotype,
            target_concentration_group=assess_concentration,
            feature=primary_feature,
            outlier_method=outlier_method,
            batch_feature=batch_feature,
            consistency_features=tuple(assess_features),
        )
        artifacts = build_cached_sensor_assessment_artifacts(filtered_features, selection)
        if artifacts.batch_error:
            logger.warning("Batch variance error: %s", artifacts.batch_error)
            st.error(f"Batch variance error: {artifacts.batch_error}")
        else:
            render_dataframe_stretch(artifacts.display_batch_table)
            if not artifacts.display_deviating_sensors_table.empty:
                st.markdown(f"**Deviating sensors (|z| > {BATCH_DEVIATION_Z_THRESHOLD:g})**")
                render_dataframe_stretch(artifacts.display_deviating_sensors_table)

            fig_batch = plot_sensor_batch_stability(
                artifacts.assessment_df,
                artifacts.display_batch_table,
                artifacts.display_batch_feature,
                sensor_col="sensor_id",
                z_threshold=BATCH_DEVIATION_Z_THRESHOLD,
            )
            render_figure_stretch(fig_batch)
    else:
        st.info("No sensor_id column; batch analysis requires sensor identifiers.")

    st.markdown("---")
    st.markdown("#### PDF Report")

    # Report consistency/batch for the top selection but the degradation trend for
    # its own (feature, concentration) so the PDF mirrors the on-screen views.
    pdf_artifacts = replace(
        artifacts,
        degradation_input_df=deg_artifacts.degradation_input_df,
        degradation_table=deg_artifacts.degradation_table,
    )

    def _generate_assessment_pdf_bytes() -> bytes:
        return build_sensor_assessment_pdf_bytes(pdf_artifacts, degradation_feature=deg_feature)

    render_pdf_download_section(
        session_key="assessment_pdf",
        filename="sensor_qc_report.pdf",
        generate_callback=_generate_assessment_pdf_bytes,
        button_label="Generate report",
        download_label="Download PDF",
    )
