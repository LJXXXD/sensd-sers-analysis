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
from .peak_discovery import (
    plot_peak_anchor_summary,
    plot_signal_level_peak_verification,
)
from .plots import VARIANCE_OPTIONS, plot_spectra
from .stats import (
    compute_feature_log_concentration_correlations,
    plot_feature_correlation_heatmap,
    plot_feature_distribution,
    plot_feature_log_concentration_correlation_bars,
    plot_feature_log_concentration_scatter,
)
from .targeted_peak_plots import (
    plot_targeted_mean_spectrum_markers,
    plot_targeted_signal_verification,
)

__all__ = [
    "VARIANCE_OPTIONS",
    "plot_batch_boxplot",
    "plot_concentration_regression",
    "plot_degradation_trend",
    "compute_feature_log_concentration_correlations",
    "plot_feature_correlation_heatmap",
    "plot_feature_distribution",
    "plot_feature_log_concentration_correlation_bars",
    "plot_feature_log_concentration_scatter",
    "plot_peak_anchor_summary",
    "plot_macro_batch_regression",
    "plot_multi_sensor_regression",
    "plot_signal_level_peak_verification",
    "plot_spectra",
    "plot_targeted_mean_spectrum_markers",
    "plot_targeted_signal_verification",
]
