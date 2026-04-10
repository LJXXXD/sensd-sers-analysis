"""
Application-layer orchestration helpers for non-UI entrypoints.
"""

from .assessment_service import (
    build_sensor_assessment_artifacts,
    build_sensor_assessment_pdf_bytes,
)
from .classification_service import (
    build_phase2_dataset,
    build_phase2_pdf_bytes,
    run_phase2_classification,
)
from .contracts import (
    DerivedDataBundle,
    FilterCatalog,
    FilterSelection,
    FilteredBundle,
    GlobalQaArtifacts,
    LoadedDataBundle,
    ModelConsistencySelection,
    OverlayArtifact,
    PeakArtifacts,
    Phase2Artifacts,
    SensorAssessmentArtifacts,
    SensorAssessmentSelection,
    SingleSensorConsistencyArtifacts,
)
from .dataset_pipeline import build_derived_bundle, load_uploaded_bundle
from .filtering_service import (
    FilterState,
    apply_filters,
    build_filter_catalog,
    compute_filter_options,
    deserialize_filter_state,
    normalize_filter_state,
    serialize_filter_state,
)
from .sensor_assessment_service import (
    build_global_qa_artifacts,
    build_overlay_artifacts,
    build_phase1_pdf_bytes,
    build_single_sensor_consistency_artifacts,
)
from .peak_discovery_service import (
    PeakAnchorOverview,
    PeakDiagnosticContext,
    PeakSignalOptions,
    SignalVerificationArtifact,
    build_matching_signal_options,
    build_peak_anchor_overviews,
    build_peak_anchor_table,
    build_peak_diagnostic_context,
    build_signal_selection_options,
    build_signal_verification_artifact,
)
from .targeted_peak_service import merge_targeted_peaks_into_filtered_bundle

__all__ = [
    "DerivedDataBundle",
    "FilterCatalog",
    "FilterSelection",
    "FilterState",
    "FilteredBundle",
    "GlobalQaArtifacts",
    "LoadedDataBundle",
    "ModelConsistencySelection",
    "OverlayArtifact",
    "PeakAnchorOverview",
    "PeakArtifacts",
    "PeakDiagnosticContext",
    "PeakSignalOptions",
    "Phase2Artifacts",
    "SensorAssessmentArtifacts",
    "SensorAssessmentSelection",
    "SignalVerificationArtifact",
    "SingleSensorConsistencyArtifacts",
    "apply_filters",
    "build_derived_bundle",
    "build_filter_catalog",
    "build_global_qa_artifacts",
    "build_matching_signal_options",
    "build_overlay_artifacts",
    "build_peak_anchor_overviews",
    "build_peak_anchor_table",
    "build_peak_diagnostic_context",
    "build_phase1_pdf_bytes",
    "build_phase2_dataset",
    "build_phase2_pdf_bytes",
    "build_sensor_assessment_artifacts",
    "build_sensor_assessment_pdf_bytes",
    "build_signal_selection_options",
    "build_signal_verification_artifact",
    "build_single_sensor_consistency_artifacts",
    "compute_filter_options",
    "deserialize_filter_state",
    "load_uploaded_bundle",
    "merge_targeted_peaks_into_filtered_bundle",
    "normalize_filter_state",
    "run_phase2_classification",
    "serialize_filter_state",
]
