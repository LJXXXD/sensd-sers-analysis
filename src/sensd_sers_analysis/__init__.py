"""
SENSD - SERS Sensor Data Analysis Package

Data loading, feature extraction, and spectral visualization for
Surface-Enhanced Raman Spectroscopy (SERS) biosensor data.
"""

from .data import (
    load_sers_data,
    get_signals_matrix,
    get_raman_shift,
    get_metadata_columns,
    wide_to_tidy,
)
from .visualization import plot_feature_distribution, plot_spectra

__version__ = "0.1.0"

__all__ = [
    "load_sers_data",
    "get_signals_matrix",
    "get_raman_shift",
    "get_metadata_columns",
    "wide_to_tidy",
    "plot_spectra",
    "plot_feature_distribution",
    "__version__",
]
