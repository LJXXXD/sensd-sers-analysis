"""
SERS signal processing: feature extraction and preprocessing.
"""

from .features import BASIC_FEATURE_COLUMNS, extract_basic_features
from .filters import (
    DEFAULT_FILTER_ORDER,
    filter_sers_data,
    get_feature_metadata_columns,
    get_filter_options,
    get_filterable_columns,
    get_plot_hue_columns,
    pick_preferred_column,
)
from .metadata import (
    add_concentration_group,
    add_log_concentration,
    preprocess_metadata,
)

__all__ = [
    "add_concentration_group",
    "add_log_concentration",
    "BASIC_FEATURE_COLUMNS",
    "DEFAULT_FILTER_ORDER",
    "extract_basic_features",
    "filter_sers_data",
    "get_feature_metadata_columns",
    "get_filter_options",
    "get_filterable_columns",
    "get_plot_hue_columns",
    "pick_preferred_column",
    "preprocess_metadata",
]
