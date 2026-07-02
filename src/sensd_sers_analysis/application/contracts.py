"""
Typed application-layer contracts for Streamlit/backend orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sensd_sers_analysis.data.io import SersLoadReport

from sensd_sers_analysis.assessment.sensor_assessment_regression import (
    CleanedRegressionResult,
)
from sensd_sers_analysis.classification.models import ClassificationResult
from sensd_sers_analysis.processing.peak_features import PeakWindowInfo
from sensd_sers_analysis.regression.models_global import SingleRegressorResult
from sensd_sers_analysis.regression.models_mtl import MtlRegressionOutputs
from sensd_sers_analysis.regression.models_two_stage import TwoStageRegressionOutputs


@dataclass(slots=True)
class LoadedDataBundle:
    """
    Parsed uploaded data before app-specific derivations.

    Parameters
    ----------
    wide_df:
        Wide-format dataframe with metadata and `rs_*` columns.
    tidy_df:
        Tidy-format dataframe derived directly from the uploaded files.
    load_report:
        Per-file load outcomes for user-facing upload feedback.
    """

    wide_df: pd.DataFrame
    tidy_df: pd.DataFrame
    load_report: SersLoadReport = field(default_factory=SersLoadReport)


@dataclass(slots=True)
class PeakArtifacts:
    """
    Dynamic peak-discovery artifacts for exploratory visualization tabs.

    Parameters
    ----------
    peak_infos_by_serotype:
        Mapping of serotype to peak-window metadata (search windows, centers).
    mean_spec_by_serotype:
        Mapping of serotype to mean spectrum used during anchor discovery.
    default_serotype:
        Fallback serotype for 0 CFU rows.
    raman_x:
        Raman-shift grid associated with the dynamic peak discovery UI.
    """

    peak_infos_by_serotype: dict[str, list[PeakWindowInfo]]
    mean_spec_by_serotype: dict[str, np.ndarray]
    default_serotype: str | None
    raman_x: np.ndarray

    @property
    def is_empty(self) -> bool:
        """Return True when no peak artifacts are available."""
        return not self.peak_infos_by_serotype or self.raman_x.size == 0


@dataclass(slots=True)
class DerivedDataBundle:
    """
    Fully derived data bundle used by the Streamlit app.

    Parameters
    ----------
    wide_df:
        Preprocessed and optionally trimmed wide-format dataframe.
    tidy_df:
        Tidy dataframe rebuilt from the trimmed wide dataframe.
    features_df:
        Feature dataframe used by downstream analysis tabs.
    peak_df:
        Peak-height dataframe returned by dynamic peak extraction.
    peak_artifacts:
        Shared peak metadata for exploratory peak discovery (not ML features).
    """

    wide_df: pd.DataFrame
    tidy_df: pd.DataFrame
    features_df: pd.DataFrame
    peak_df: pd.DataFrame
    peak_artifacts: PeakArtifacts


@dataclass(frozen=True, slots=True)
class FilterSelection:
    """
    Canonical representation of one UI filter.

    Parameters
    ----------
    selected_values:
        Selected values for the filter dimension. An empty tuple means no filter.
    exclude:
        If True, treat `selected_values` as an exclusion list.
    """

    selected_values: tuple[str, ...] = ()
    exclude: bool = False

    def as_processing_state(self) -> tuple[list[str] | None, bool]:
        """
        Convert to the legacy `(selected, exclude)` format.

        Returns
        -------
        tuple[list[str] | None, bool]
            Existing processing-layer state format.
        """

        if not self.selected_values:
            return None, self.exclude
        return list(self.selected_values), self.exclude


@dataclass(frozen=True, slots=True)
class FilterCatalog:
    """
    Ordered filter catalog for the Streamlit sidebar.

    Parameters
    ----------
    filter_columns:
        All filterable columns in display order.
    main_columns:
        First group of prominently displayed filters.
    more_columns:
        Remaining filters shown in the expander.
    """

    filter_columns: tuple[str, ...]
    main_columns: tuple[str, ...]
    more_columns: tuple[str, ...]


@dataclass(slots=True)
class FilteredBundle:
    """
    Filtered views derived from a `DerivedDataBundle`.

    Parameters
    ----------
    filtered_tidy_df:
        Filtered tidy dataframe for spectra plotting.
    filtered_features_df:
        Filtered feature dataframe for analysis tabs.
    n_unique_spectra:
        Count of unique `(filename, signal_index)` pairs in the tidy view.
    """

    filtered_tidy_df: pd.DataFrame
    filtered_features_df: pd.DataFrame
    n_unique_spectra: int


@dataclass(frozen=True, slots=True)
class SensorAssessmentSelection:
    """
    User selection for the sensor assessment workflow.

    Parameters
    ----------
    serotype:
        Selected serotype.
    concentration_group:
        Selected concentration group.
    feature:
        Primary feature used for consistency and degradation analysis.
    outlier_method:
        Outlier method used for consistency calculations.
    batch_feature:
        Feature used for displayed batch analysis.
    """

    serotype: str
    concentration_group: str
    feature: str
    outlier_method: str
    batch_feature: str


@dataclass(slots=True)
class SensorAssessmentArtifacts:
    """
    Precomputed sensor-assessment artifacts for display and PDF export.

    Parameters
    ----------
    assessment_df:
        Filtered dataframe restricted to the selected serotype/concentration.
    consistency_group_cols:
        Grouping columns used for consistency summaries.
    display_consistency_table:
        Consistency table shown in the tab.
    pdf_consistency_table:
        Consistency table preserved for PDF parity.
    degradation_input_df:
        Prepared degradation dataframe with `test_ordinal`.
    degradation_table:
        Degradation summary table for the selected feature.
    display_batch_feature:
        Batch feature selected in the UI.
    display_batch_table:
        Batch variance table shown in the tab.
    display_deviating_sensors_table:
        Deviating-sensor table shown in the tab.
    pdf_batch_table:
        Batch variance table preserved for PDF parity.
    pdf_deviating_sensors_table:
        Deviating-sensor table preserved for PDF parity.
    selection:
        Original assessment selection.
    consistency_error:
        Consistency-section error message, if any.
    degradation_error:
        Degradation-section error message, if any.
    batch_error:
        Batch-analysis error message, if any.
    """

    assessment_df: pd.DataFrame
    consistency_group_cols: list[str] | None
    display_consistency_table: pd.DataFrame
    pdf_consistency_table: pd.DataFrame
    degradation_input_df: pd.DataFrame
    degradation_table: pd.DataFrame
    display_batch_feature: str
    display_batch_table: pd.DataFrame
    display_deviating_sensors_table: pd.DataFrame
    pdf_batch_table: pd.DataFrame
    pdf_deviating_sensors_table: pd.DataFrame
    selection: SensorAssessmentSelection
    consistency_error: str | None = None
    degradation_error: str | None = None
    batch_error: str | None = None


@dataclass(frozen=True, slots=True)
class ModelConsistencySelection:
    """
    User selection for single-sensor regression QA (sensor assessment).

    Parameters
    ----------
    sensor_id:
        Selected sensor identifier.
    serotype:
        Selected serotype.
    feature:
        Feature used for regression QA.
    """

    sensor_id: str
    serotype: str
    feature: str


@dataclass(slots=True)
class SingleSensorConsistencyArtifacts:
    """
    Precomputed artifacts for one sensor/serotype/feature regression view.

    Parameters
    ----------
    model_df:
        Subset dataframe used for the regression analysis.
    regression_result:
        Two-pass cleaned regression result, if available.
    zero_cfu_baseline:
        Mean zero-CFU baseline for the selected feature.
    selection:
        Original user selection.
    """

    model_df: pd.DataFrame
    regression_result: CleanedRegressionResult | None
    zero_cfu_baseline: float | None
    selection: ModelConsistencySelection


@dataclass(slots=True)
class GlobalQaArtifacts:
    """
    Global model-consistency QA outputs.

    Parameters
    ----------
    table:
        QA summary table.
    excluded_map:
        Mapping of `(serotype, feature)` to excluded sensor identifiers.
    selected_features:
        Features used to populate the QA table.
    """

    table: pd.DataFrame
    excluded_map: dict[tuple[str, str], set[str]]
    selected_features: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OverlayArtifact:
    """
    Overlay/macro-regression request for one serotype-feature pair.

    Parameters
    ----------
    serotype:
        Serotype displayed in the overlay.
    feature:
        Feature displayed in the overlay.
    excluded_sensors:
        Sensors excluded by the global QA policy.
    pass_sensors:
        Sensors that remain after exclusion.
    """

    serotype: str
    feature: str
    excluded_sensors: frozenset[str]
    pass_sensors: frozenset[str]


@dataclass(slots=True)
class GlobalRegressionArtifacts:
    """
    Paradigm 1: pooled global regressors evaluated on held-out sensors.

    Parameters
    ----------
    regression_clean:
        Clean positive-CFU rows (dynamic serotypes) used for training and evaluation.
    feature_columns:
        Feature names passed to the models.
    train_indices, test_indices:
        Positional indices after ``reset_index`` on ``regression_clean``.
    rf_result, svm_result:
        Random Forest and SVR results on the test split.
    best_result:
        Model with lower test RMSE.
    """

    regression_clean: pd.DataFrame
    feature_columns: tuple[str, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray
    rf_result: SingleRegressorResult
    svm_result: SingleRegressorResult
    best_result: SingleRegressorResult


@dataclass(slots=True)
class TwoStageRegressionArtifacts:
    """
    Paradigm 2: multi-class serotype routing + serotype-specific regressors.
    """

    regression_clean: pd.DataFrame
    feature_columns: tuple[str, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray
    outputs: TwoStageRegressionOutputs


@dataclass(slots=True)
class MtlRegressionArtifacts:
    """
    Paradigm 3: shared-trunk multi-task network outputs.
    """

    regression_clean: pd.DataFrame
    feature_columns: tuple[str, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray
    outputs: MtlRegressionOutputs


@dataclass(frozen=True, slots=True)
class ClassificationArtifacts:
    """
    Outputs from serotype classification training (RF / SVM).

    Parameters
    ----------
    clean_classification_df:
        Cleaned dataframe used for classification.
    feature_columns:
        Feature columns passed to the classifiers.
    rf_result:
        Random Forest result bundle.
    svm_result:
        SVM result bundle.
    best_result:
        Result chosen as the best-performing model.
    """

    clean_classification_df: pd.DataFrame
    feature_columns: tuple[str, ...]
    rf_result: ClassificationResult
    svm_result: ClassificationResult
    best_result: ClassificationResult
