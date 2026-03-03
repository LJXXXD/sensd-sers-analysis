"""
SERS signal processing: feature extraction and preprocessing.
"""

from .alignment import trim_raman_shift
from .features import (
    BASIC_FEATURE_COLUMNS,
    DEFAULT_GLOBAL_QA_FEATURES,
    PREFERRED_FEATURE_ORDER,
    PHASE2_FEATURE_BASE,
    extract_basic_features,
    get_available_feature_columns,
    order_features_by_preference,
)
from .filters import (
    DEFAULT_FILTER_ORDER,
    filter_by_selections,
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
from .pca_features import add_pca_features
from .peak_features import (
    PeakWindowInfo,
    extract_dynamic_peak_features,
    get_peak_height_columns,
)

__all__ = [
    "add_concentration_group",
    "add_pca_features",
    "BASIC_FEATURE_COLUMNS",
    "DEFAULT_GLOBAL_QA_FEATURES",
    "get_available_feature_columns",
    "order_features_by_preference",
    "PHASE2_FEATURE_BASE",
    "PREFERRED_FEATURE_ORDER",
    "PeakWindowInfo",
    "extract_dynamic_peak_features",
    "get_peak_height_columns",
    "add_log_concentration",
    "DEFAULT_FILTER_ORDER",
    "extract_basic_features",
    "trim_raman_shift",
    "filter_by_selections",
    "filter_sers_data",
    "get_feature_metadata_columns",
    "get_filter_options",
    "get_filterable_columns",
    "get_plot_hue_columns",
    "pick_preferred_column",
    "preprocess_metadata",
]
