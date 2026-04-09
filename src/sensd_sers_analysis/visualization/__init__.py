"""
SERS spectral and statistical plotting.
"""

from .assessment_plots import (
    plot_batch_boxplot,
    plot_concentration_regression,
    plot_degradation_trend,
    plot_macro_batch_regression,
    plot_multi_sensor_regression,
)
from .peak_diagnostics import (
    plot_peak_anchor_summary,
    plot_signal_level_peak_verification,
)
from .plots import VARIANCE_OPTIONS, plot_spectra
from .stats import plot_feature_distribution

__all__ = [
    "VARIANCE_OPTIONS",
    "plot_batch_boxplot",
    "plot_concentration_regression",
    "plot_degradation_trend",
    "plot_feature_distribution",
    "plot_peak_anchor_summary",
    "plot_macro_batch_regression",
    "plot_multi_sensor_regression",
    "plot_signal_level_peak_verification",
    "plot_spectra",
]
