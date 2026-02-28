"""
SERS spectral and statistical plotting.
"""

from .assessment_plots import plot_batch_boxplot, plot_degradation_trend
from .plots import plot_spectra
from .stats import plot_feature_distribution

__all__ = [
    "plot_batch_boxplot",
    "plot_degradation_trend",
    "plot_feature_distribution",
    "plot_spectra",
]
