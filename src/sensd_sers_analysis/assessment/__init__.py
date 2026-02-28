"""
Sensor assessment: consistency, degradation, and batch stability.

Provides statistical infrastructure for SERS sensor evaluation without
hardcoding biological thresholds.
"""

from .batch_variance import (
    compute_batch_variance,
    identify_deviating_sensors,
)
from .consistency import (
    coefficient_of_variation,
    compute_consistency_metrics,
    get_consistency_summary_table,
)
from .degradation import (
    add_sequence_column,
    compute_degradation,
    prepare_degradation_data,
)
from .outliers import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    filter_outliers,
)

__all__ = [
    "add_sequence_column",
    "coefficient_of_variation",
    "compute_batch_variance",
    "compute_consistency_metrics",
    "compute_degradation",
    "prepare_degradation_data",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "filter_outliers",
    "get_consistency_summary_table",
    "identify_deviating_sensors",
]
