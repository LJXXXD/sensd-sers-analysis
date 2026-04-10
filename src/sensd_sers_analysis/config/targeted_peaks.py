"""
Configuration for fixed-anchor (targeted) peak feature extraction.
"""

from __future__ import annotations

# Default number of targeted peak features shown in the Streamlit UI.
TARGETED_PEAK_DEFAULT_COUNT: int = 5

# Default Raman-shift anchors (cm⁻¹) for targeted peak-height features.
TARGETED_PEAK_DEFAULT_ANCHORS_CM1: tuple[float, ...] = (
    501.8,
    613.7,
    809.7,
    1066.5,
    1196.8,
)

# Half-width (cm⁻¹) of the search interval around each anchor: [a−w, a+w].
TARGETED_PEAK_SEARCH_HALF_WIDTH_CM1: float = 10.0

__all__ = [
    "TARGETED_PEAK_DEFAULT_ANCHORS_CM1",
    "TARGETED_PEAK_DEFAULT_COUNT",
    "TARGETED_PEAK_SEARCH_HALF_WIDTH_CM1",
]
