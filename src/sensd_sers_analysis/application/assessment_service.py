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
    filter_outliers,
    get_consistency_summary_table,
    identify_deviating_sensors,
    prepare_degradation_data,
)
from sensd_sers_analysis.config import BATCH_DEVIATION_Z_THRESHOLD
from sensd_sers_analysis.processing import filter_by_selections
from sensd_sers_analysis.report import build_sensor_assessment_pdf
from sensd_sers_analysis.visualization.assessment_plots import (
    plot_degradation_trend,
    plot_sensor_batch_stability,
)


def _clean_degradation_replicates(
    df: pd.DataFrame,
    feature: str,
    outlier_method: str,
) -> pd.DataFrame:
    """
    Remove outlier reads within each test session before the degradation fit.

    Outliers are dropped *within* each ``(sensor_id, test_id)`` group so aberrant
    individual reads do not distort a session's mean. Filtering is deliberately
    scoped to within-session replicates and never across time, so a genuine
    across-test jump (e.g. a contamination/carryover spike) is preserved in the
    trend rather than silently removed.

    Parameters
    ----------
    df:
        Assessment dataframe already restricted to one serotype/target.
    feature:
        Feature column used for outlier detection.
    outlier_method:
        ``"iqr"`` or ``"zscore"``.

    Returns
    -------
    pd.DataFrame
        Rows surviving within-session outlier removal.
    """

    if feature not in df.columns or df.empty:
        return df
    group_cols = [c for c in ("sensor_id", "test_id") if c in df.columns]
    if not group_cols:
        inliers, _ = filter_outliers(df, feature, method=outlier_method)
        return inliers

    cleaned_parts = [
        filter_outliers(group, feature, method=outlier_method)[0]
        for _, group in df.groupby(group_cols, dropna=False)
    ]
    return pd.concat(cleaned_parts) if cleaned_parts else df


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
            "target_concentration_group": selection.target_concentration_group,
        },
    )
    consistency_group_cols = [
        column for column in ASSESSMENT_GROUP_COLS if column in assessment_df.columns
    ]
    if not consistency_group_cols:
        consistency_group_cols = ["sensor_id"] if "sensor_id" in assessment_df.columns else None

    display_feature_cols = list(selection.consistency_features) or [selection.feature]

    display_consistency_table = pd.DataFrame()
    pdf_consistency_table = pd.DataFrame()
    consistency_error = None
    try:
        display_consistency_table = get_consistency_summary_table(
            assessment_df,
            feature_cols=display_feature_cols,
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
        degradation_source = _clean_degradation_replicates(
            assessment_df,
            selection.feature,
            selection.outlier_method,
        )
        degradation_input_df = prepare_degradation_data(
            degradation_source,
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
    *,
    degradation_feature: str | None = None,
) -> bytes:
    """
    Build the sensor assessment PDF from precomputed artifacts.

    Parameters
    ----------
    artifacts:
        Precomputed artifacts returned by `build_sensor_assessment_artifacts`.
    degradation_feature:
        Feature column plotted for the degradation trend. Defaults to
        ``artifacts.selection.feature``; pass an explicit value when the
        degradation view uses a feature different from the consistency selection.

    Returns
    -------
    bytes
        PDF document bytes.
    """

    degradation_column = degradation_feature or artifacts.selection.feature
    degradation_fig = None
    if not artifacts.degradation_input_df.empty and len(artifacts.degradation_input_df) >= 2:
        degradation_fig = plot_degradation_trend(
            artifacts.degradation_input_df,
            degradation_column,
            "test_ordinal",
            group_col=(
                "sensor_id" if "sensor_id" in artifacts.degradation_input_df.columns else None
            ),
        )

    batch_fig = None
    if (
        "sensor_id" in artifacts.assessment_df.columns
        and not artifacts.assessment_df.empty
        and not artifacts.pdf_batch_table.empty
    ):
        batch_fig = plot_sensor_batch_stability(
            artifacts.assessment_df,
            artifacts.pdf_batch_table,
            artifacts.selection.feature,
            sensor_col="sensor_id",
            z_threshold=BATCH_DEVIATION_Z_THRESHOLD,
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
            f"{artifacts.selection.serotype}, "
            f"{artifacts.selection.target_concentration_group}"
        ),
    )
