"""
SENSD - SERS Sensor Data Analysis Package

Data loading, feature extraction, spectral visualization, and sensor
assessment for Surface-Enhanced Raman Spectroscopy (SERS) biosensor data.
"""

from .assessment import (
    add_sequence_column,
    coefficient_of_variation,
    compute_batch_variance,
    compute_consistency_metrics,
    compute_degradation,
    detect_outliers_iqr,
    detect_outliers_zscore,
    filter_outliers,
    get_consistency_summary_table,
    identify_deviating_sensors,
)
from .data import (
    get_metadata_columns,
    get_raman_shift,
    get_signals_matrix,
    load_sers_data,
    wide_to_tidy,
)
from .report import build_sensor_assessment_pdf
from .visualization import (
    plot_batch_boxplot,
    plot_degradation_trend,
    plot_feature_distribution,
    plot_spectra,
)

__version__ = "0.1.0"

__all__ = [
    "add_sequence_column",
    "build_sensor_assessment_pdf",
    "coefficient_of_variation",
    "compute_batch_variance",
    "compute_consistency_metrics",
    "compute_degradation",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "filter_outliers",
    "get_consistency_summary_table",
    "get_metadata_columns",
    "get_raman_shift",
    "get_signals_matrix",
    "identify_deviating_sensors",
    "load_sers_data",
    "plot_batch_boxplot",
    "plot_degradation_trend",
    "plot_feature_distribution",
    "plot_spectra",
    "wide_to_tidy",
    "__version__",
]
