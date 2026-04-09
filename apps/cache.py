"""
Streamlit cache wrappers around application-layer services.
"""

from __future__ import annotations

import streamlit as st

from sensd_sers_analysis.application.assessment_service import (
    build_sensor_assessment_artifacts,
)
from sensd_sers_analysis.application.classification_service import (
    build_phase2_dataset,
    run_phase2_classification,
)
from sensd_sers_analysis.application.contracts import SensorAssessmentSelection
from sensd_sers_analysis.application.dataset_pipeline import build_derived_bundle
from sensd_sers_analysis.application.filtering_service import deserialize_filter_state
from sensd_sers_analysis.application.model_consistency_service import (
    build_global_qa_artifacts,
    build_single_sensor_consistency_artifacts,
)


@st.cache_data
def build_cached_derived_bundle(
    loaded_bundle,
    *,
    min_shift: float | None,
    max_shift: float | None,
    n_peaks: int,
    n_peaks_by_serotype_items: tuple[tuple[str, int], ...],
):
    """
    Cache the derived dataset bundle for a fixed set of sidebar controls.

    Parameters
    ----------
    loaded_bundle:
        Parsed upload bundle.
    min_shift:
        Lower Raman-shift trim bound.
    max_shift:
        Upper Raman-shift trim bound.
    n_peaks:
        Default peak count.
    n_peaks_by_serotype_items:
        Serialized serotype-specific peak-count mapping.

    Returns
    -------
    DerivedDataBundle
        Cached derived dataset bundle.
    """

    n_peaks_by_serotype = dict(n_peaks_by_serotype_items) if n_peaks_by_serotype_items else None
    return build_derived_bundle(
        loaded_bundle,
        min_shift=min_shift,
        max_shift=max_shift,
        n_peaks=n_peaks,
        n_peaks_by_serotype=n_peaks_by_serotype,
    )


@st.cache_data
def apply_cached_filters(derived_bundle, serialized_filter_state):
    """
    Cache filtered tidy/features views for a serialized filter state.

    Parameters
    ----------
    derived_bundle:
        Derived application data bundle.
    serialized_filter_state:
        Serialized filter-state representation.

    Returns
    -------
    FilteredBundle
        Cached filtered view bundle.
    """

    from sensd_sers_analysis.application import apply_filters

    filter_state = deserialize_filter_state(serialized_filter_state)
    return apply_filters(derived_bundle, filter_state)


@st.cache_data
def build_cached_sensor_assessment_artifacts(
    filtered_features,
    selection: SensorAssessmentSelection,
):
    """
    Cache sensor-assessment artifacts for a specific user selection.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe.
    selection:
        Sensor-assessment selection.

    Returns
    -------
    SensorAssessmentArtifacts
        Cached sensor-assessment artifacts.
    """

    return build_sensor_assessment_artifacts(filtered_features, selection)


@st.cache_data
def build_cached_single_sensor_consistency_artifacts(filtered_features, selection):
    """
    Cache single-sensor model-consistency artifacts.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe.
    selection:
        Model-consistency selection.

    Returns
    -------
    SingleSensorConsistencyArtifacts
        Cached single-sensor regression artifacts.
    """

    return build_single_sensor_consistency_artifacts(filtered_features, selection)


@st.cache_data
def build_cached_global_qa_artifacts(filtered_features, feature_columns: tuple[str, ...]):
    """
    Cache global model-consistency QA artifacts.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe.
    feature_columns:
        Features selected for the global QA table.

    Returns
    -------
    GlobalQaArtifacts
        Cached QA table and excluded-sensor map.
    """

    return build_global_qa_artifacts(filtered_features, feature_columns)


@st.cache_data
def build_cached_phase2_dataset(
    filtered_features,
    *,
    excluded_map_policy: tuple[str, ...],
    inlier_feature: str,
):
    """
    Cache Phase 2 clean-data preparation.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe.
    excluded_map_policy:
        Features used to build the exclusion map.
    inlier_feature:
        Feature used for Phase 2 inlier filtering.

    Returns
    -------
    pd.DataFrame
        Cached Phase 2 clean dataframe.
    """

    return build_phase2_dataset(
        filtered_features,
        excluded_map_policy=excluded_map_policy,
        inlier_feature=inlier_feature,
    )


@st.cache_data
def build_cached_phase2_artifacts(phase2_clean, feature_columns: tuple[str, ...]):
    """
    Cache Phase 2 model training outputs.

    Parameters
    ----------
    phase2_clean:
        Clean Phase 2 dataframe.
    feature_columns:
        Feature columns used for classification.

    Returns
    -------
    Phase2Artifacts
        Cached classification outputs.
    """

    return run_phase2_classification(phase2_clean, feature_columns)
