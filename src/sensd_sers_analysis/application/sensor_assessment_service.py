"""
Application-layer orchestration for sensor assessment (regression QA) workflows.
"""

from __future__ import annotations

import pandas as pd

from sensd_sers_analysis.application.contracts import (
    GlobalQaArtifacts,
    ModelConsistencySelection,
    OverlayArtifact,
    SingleSensorConsistencyArtifacts,
)
from sensd_sers_analysis.assessment import (
    fit_concentration_regression_cleaned,
    get_global_model_consistency_qa,
    get_zero_cfu_baseline,
)
from sensd_sers_analysis.processing import filter_by_selections
from sensd_sers_analysis.report import build_sensor_assessment_qa_pdf
from sensd_sers_analysis.visualization.assessment_plots import (
    plot_macro_batch_regression,
    plot_multi_sensor_regression,
)


def build_single_sensor_consistency_artifacts(
    filtered_features: pd.DataFrame,
    selection: ModelConsistencySelection,
) -> SingleSensorConsistencyArtifacts:
    """
    Build artifacts for the single-sensor regression QA view.

    Parameters
    ----------
    filtered_features:
        Feature dataframe after global app filters are applied.
    selection:
        User selection for sensor, serotype, and feature.

    Returns
    -------
    SingleSensorConsistencyArtifacts
        Precomputed single-sensor regression inputs and outputs.
    """

    model_df = filter_by_selections(
        filtered_features,
        {
            "sensor_id": selection.sensor_id,
            "serotype": selection.serotype,
        },
    )
    regression_result = None
    zero_cfu_baseline = None
    if not model_df.empty:
        regression_result = fit_concentration_regression_cleaned(model_df, selection.feature)
        zero_cfu_baseline = get_zero_cfu_baseline(model_df, selection.feature)

    return SingleSensorConsistencyArtifacts(
        model_df=model_df,
        regression_result=regression_result,
        zero_cfu_baseline=zero_cfu_baseline,
        selection=selection,
    )


def build_global_qa_artifacts(
    filtered_features: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> GlobalQaArtifacts:
    """
    Build global QA artifacts for the model-consistency tab.

    Parameters
    ----------
    filtered_features:
        Feature dataframe after global app filters are applied.
    feature_columns:
        Features selected for the QA table.

    Returns
    -------
    GlobalQaArtifacts
        Global QA table and excluded-sensor mapping.
    """

    if not feature_columns:
        return GlobalQaArtifacts(table=pd.DataFrame(), excluded_map={}, selected_features=())

    table, excluded_map = get_global_model_consistency_qa(
        filtered_features,
        feature_cols=list(feature_columns),
    )
    return GlobalQaArtifacts(
        table=table,
        excluded_map=excluded_map,
        selected_features=feature_columns,
    )


def build_overlay_artifacts(
    filtered_features: pd.DataFrame,
    overlay_serotypes: tuple[str, ...],
    overlay_features: tuple[str, ...],
    excluded_map: dict[tuple[str, str], set[str]],
) -> list[OverlayArtifact]:
    """
    Build overlay requests for multi-sensor regression and macro regression.

    Parameters
    ----------
    filtered_features:
        Feature dataframe after global app filters are applied.
    overlay_serotypes:
        Serotypes selected for the overlay section.
    overlay_features:
        Features selected for the overlay section.
    excluded_map:
        Excluded-sensor mapping from the global QA workflow.

    Returns
    -------
    list[OverlayArtifact]
        Overlay requests with resolved excluded/pass sensor sets.
    """

    overlay_artifacts: list[OverlayArtifact] = []
    for serotype in overlay_serotypes:
        sensor_series = filtered_features.loc[
            filtered_features["serotype"].astype(str) == str(serotype),
            "sensor_id",
        ]
        all_sensors = frozenset(sensor_series.dropna().astype(str).unique().tolist())
        for feature in overlay_features:
            excluded_sensors = frozenset(
                str(sensor_id)
                for sensor_id in excluded_map.get((str(serotype), str(feature)), set())
            )
            pass_sensors = frozenset(sorted(all_sensors - excluded_sensors))
            overlay_artifacts.append(
                OverlayArtifact(
                    serotype=str(serotype),
                    feature=str(feature),
                    excluded_sensors=excluded_sensors,
                    pass_sensors=pass_sensors,
                )
            )
    return overlay_artifacts


def build_sensor_assessment_qa_pdf_bytes(
    filtered_features: pd.DataFrame,
    global_qa_artifacts: GlobalQaArtifacts,
    overlay_artifacts: list[OverlayArtifact],
    *,
    report_title: str,
) -> bytes:
    """
    Build the sensor assessment QA PDF from cached artifacts.

    Parameters
    ----------
    filtered_features:
        Feature dataframe after global app filters are applied.
    global_qa_artifacts:
        Cached global QA results.
    overlay_artifacts:
        Cached overlay requests.
    report_title:
        Title used in the generated PDF.

    Returns
    -------
    bytes
        PDF document bytes.
    """

    overlay_items: list[dict] = []
    macro_items: list[dict] = []
    for artifact in overlay_artifacts:
        try:
            overlay_fig = plot_multi_sensor_regression(
                filtered_features,
                artifact.serotype,
                artifact.feature,
                excluded_sensors=set(artifact.excluded_sensors),
            )
            overlay_items.append(
                {
                    "fig": overlay_fig,
                    "serotype": artifact.serotype,
                    "feature": artifact.feature,
                }
            )
        except ValueError:
            pass

        try:
            macro_fig, macro_result = plot_macro_batch_regression(
                filtered_features,
                artifact.serotype,
                artifact.feature,
                set(artifact.pass_sensors),
            )
            macro_items.append(
                {
                    "fig": macro_fig,
                    "macro_result": macro_result,
                    "serotype": artifact.serotype,
                    "feature": artifact.feature,
                }
            )
        except ValueError:
            pass

    return build_sensor_assessment_qa_pdf(
        global_qa_table=(
            global_qa_artifacts.table if not global_qa_artifacts.table.empty else None
        ),
        overlay_items=overlay_items,
        macro_items=macro_items,
        report_title=report_title,
    )
