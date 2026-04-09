"""
Application-layer orchestration for the sensor assessment workflow.
"""

from __future__ import annotations

import pandas as pd

from sensd_sers_analysis.application.contracts import (
    SensorAssessmentArtifacts,
    SensorAssessmentSelection,
)
from sensd_sers_analysis.assessment import (
    ASSESSMENT_GROUP_COLS,
    compute_batch_variance,
    compute_degradation,
    get_consistency_summary_table,
    identify_deviating_sensors,
    prepare_degradation_data,
)
from sensd_sers_analysis.config import BATCH_DEVIATION_Z_THRESHOLD
from sensd_sers_analysis.processing import filter_by_selections
from sensd_sers_analysis.report import build_sensor_assessment_pdf
from sensd_sers_analysis.visualization.assessment_plots import (
    plot_batch_boxplot,
    plot_degradation_trend,
)


def build_sensor_assessment_artifacts(
    filtered_features: pd.DataFrame,
    selection: SensorAssessmentSelection,
) -> SensorAssessmentArtifacts:
    """
    Build display/PDF artifacts for the sensor assessment tab.

    Parameters
    ----------
    filtered_features:
        Feature dataframe after global app filters are applied.
    selection:
        User selection for serotype, concentration, feature, and outlier policy.

    Returns
    -------
    SensorAssessmentArtifacts
        Precomputed tables and prepared dataframes used by the tab and PDF
        builder.
    """

    assessment_df = filter_by_selections(
        filtered_features,
        {
            "serotype": selection.serotype,
            "concentration_group": selection.concentration_group,
        },
    )
    consistency_group_cols = [
        column for column in ASSESSMENT_GROUP_COLS if column in assessment_df.columns
    ]
    if not consistency_group_cols:
        consistency_group_cols = ["sensor_id"] if "sensor_id" in assessment_df.columns else None

    display_consistency_table = pd.DataFrame()
    pdf_consistency_table = pd.DataFrame()
    consistency_error = None
    try:
        display_consistency_table = get_consistency_summary_table(
            assessment_df,
            feature_cols=[selection.feature],
            group_cols=consistency_group_cols,
            outlier_method=selection.outlier_method,
        )
        pdf_consistency_table = get_consistency_summary_table(
            assessment_df,
            group_cols=consistency_group_cols,
            outlier_method=selection.outlier_method,
        )
    except ValueError as exc:
        consistency_error = str(exc)

    degradation_input_df = pd.DataFrame()
    degradation_table = pd.DataFrame()
    degradation_error = None
    try:
        degradation_input_df = prepare_degradation_data(
            assessment_df,
            selection.feature,
            test_col="test_id",
            date_col="date",
        )
        if not degradation_input_df.empty and len(degradation_input_df) >= 2:
            degradation_table = compute_degradation(
                degradation_input_df,
                selection.feature,
                "test_ordinal",
                group_cols=(["sensor_id"] if "sensor_id" in degradation_input_df.columns else None),
            )
    except ValueError as exc:
        degradation_error = str(exc)

    display_batch_table = pd.DataFrame()
    display_deviating_table = pd.DataFrame()
    pdf_batch_table = pd.DataFrame()
    pdf_deviating_table = pd.DataFrame()
    batch_error = None
    if "sensor_id" in assessment_df.columns:
        try:
            display_batch_table = compute_batch_variance(
                assessment_df,
                selection.batch_feature,
                sensor_col="sensor_id",
                group_cols=None,
            )
            display_deviating_table = identify_deviating_sensors(
                display_batch_table,
                z_threshold=BATCH_DEVIATION_Z_THRESHOLD,
                sensor_col="sensor_id",
            )
            pdf_batch_table = compute_batch_variance(
                assessment_df,
                selection.feature,
                sensor_col="sensor_id",
                group_cols=None,
            )
            pdf_deviating_table = identify_deviating_sensors(
                pdf_batch_table,
                z_threshold=BATCH_DEVIATION_Z_THRESHOLD,
                sensor_col="sensor_id",
            )
        except ValueError as exc:
            batch_error = str(exc)

    return SensorAssessmentArtifacts(
        assessment_df=assessment_df,
        consistency_group_cols=consistency_group_cols,
        display_consistency_table=display_consistency_table,
        pdf_consistency_table=pdf_consistency_table,
        degradation_input_df=degradation_input_df,
        degradation_table=degradation_table,
        display_batch_feature=selection.batch_feature,
        display_batch_table=display_batch_table,
        display_deviating_sensors_table=display_deviating_table,
        pdf_batch_table=pdf_batch_table,
        pdf_deviating_sensors_table=pdf_deviating_table,
        selection=selection,
        consistency_error=consistency_error,
        degradation_error=degradation_error,
        batch_error=batch_error,
    )


def build_sensor_assessment_pdf_bytes(
    artifacts: SensorAssessmentArtifacts,
) -> bytes:
    """
    Build the sensor assessment PDF from precomputed artifacts.

    Parameters
    ----------
    artifacts:
        Precomputed artifacts returned by `build_sensor_assessment_artifacts`.

    Returns
    -------
    bytes
        PDF document bytes.
    """

    degradation_fig = None
    if not artifacts.degradation_input_df.empty and len(artifacts.degradation_input_df) >= 2:
        degradation_fig = plot_degradation_trend(
            artifacts.degradation_input_df,
            artifacts.selection.feature,
            "test_ordinal",
            group_col=(
                "sensor_id" if "sensor_id" in artifacts.degradation_input_df.columns else None
            ),
        )

    batch_fig = None
    if "sensor_id" in artifacts.assessment_df.columns and not artifacts.assessment_df.empty:
        batch_fig = plot_batch_boxplot(
            artifacts.assessment_df,
            artifacts.selection.feature,
            sensor_col="sensor_id",
            group_col=None,
        )

    return build_sensor_assessment_pdf(
        consistency_table=(
            artifacts.pdf_consistency_table if not artifacts.pdf_consistency_table.empty else None
        ),
        degradation_table=(
            artifacts.degradation_table if not artifacts.degradation_table.empty else None
        ),
        degradation_fig=degradation_fig,
        batch_variance_table=artifacts.pdf_batch_table
        if not artifacts.pdf_batch_table.empty
        else None,
        batch_boxplot_fig=batch_fig,
        deviating_sensors_table=(
            artifacts.pdf_deviating_sensors_table
            if not artifacts.pdf_deviating_sensors_table.empty
            else None
        ),
        outlier_method=artifacts.selection.outlier_method,
        report_title=(
            "SERS Sensor Assessment — "
            f"{artifacts.selection.serotype}, {artifacts.selection.concentration_group}"
        ),
    )
