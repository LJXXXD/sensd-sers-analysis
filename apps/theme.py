"""
Theme constants for SERS Data Explorer UI.

Centralizes magic numbers, figsize presets, slider limits, and standard HTML.
"""

# ---------------------------------------------------------------------------
# Streamlit display
# ---------------------------------------------------------------------------
FIGURE_WIDTH = "stretch"
DATAFRAME_WIDTH = "stretch"
HIDE_INDEX = True

# ---------------------------------------------------------------------------
# Plot dimensions (inches)
# ---------------------------------------------------------------------------
DEFAULT_FIGSIZE_WIDE = (14, 5)
DEFAULT_FIGSIZE_ANCHOR = (14, 4)
DEFAULT_FIGSIZE_WIDTH = 14

# ---------------------------------------------------------------------------
# Slider limits
# ---------------------------------------------------------------------------
PLOT_HEIGHT_MIN = 4
PLOT_HEIGHT_MAX = 12
PLOT_HEIGHT_DEFAULT = 10
N_PEAKS_MIN = 1
N_PEAKS_MAX = 10
N_PEAKS_DEFAULT = 6

# ---------------------------------------------------------------------------
# Matplotlib aesthetics
# ---------------------------------------------------------------------------
LEGEND_FONTSIZE = 8
GRID_ALPHA = 0.3
SPAN_ALPHA_ANCHOR = 0.15  # Peak window shading (mean spectrum)
SPAN_ALPHA_SIGNAL = 0.12  # Peak window shading (signal-level plot)
AXVLINE_ALPHA = 0.8

# ---------------------------------------------------------------------------
# HTML dividers
# ---------------------------------------------------------------------------
SECTION_DIVIDER_HTML = (
    '<hr style="margin: 1rem 0; padding: 0; border: 0; '
    'border-top: 3px solid currentColor; opacity: 0.5;">'
)
TITLE_TO_FILTER_DIVIDER_HTML = (
    '<hr style="margin: 0.4rem 0; padding: 0; border: 0; '
    'border-top: 1px solid currentColor; opacity: 0.25;">'
)
FILTER_DIVIDER_HTML = (
    '<hr style="margin: 0.25rem 0; padding: 0; border: 0; '
    'border-top: 1px solid currentColor; opacity: 0.2;">'
)
