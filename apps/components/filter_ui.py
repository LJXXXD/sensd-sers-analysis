"""
Filter UI components for SERS Data Explorer.
"""

import streamlit as st

from sensd_sers_analysis.utils import format_column_label

# Selector for horizontal filter header containers (key=filter_header_*).
# Scoped to sidebar only; does not affect multiselect, pills, or number inputs below.
_FILTER_HEADER_SELECTOR = '[data-testid="stSidebar"] [class*="st-key-filter_header"]'

# Selector for main Filters title + Reset All button (key=main_filter_header).
_MAIN_HEADER_SELECTOR = '[data-testid="stSidebar"] [class*="st-key-main_filter_header"]'

# Combined selector for all horizontal header blocks (filter headers + main header).
_HEADERS_SELECTOR = f"{_FILTER_HEADER_SELECTOR}, {_MAIN_HEADER_SELECTOR}"

FILTER_UI_CSS = f"""
<style>
/* 1. Main horizontal wrapper layout */
{_HEADERS_SELECTOR} [data-testid="stHorizontalBlock"] {{
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 0.75rem !important;
}}

/* 2. Target Streamlit's hidden column wrappers */
{_HEADERS_SELECTOR} [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
    display: flex !important;
    align-items: center !important;
    width: auto !important; /* Overrides Streamlit's inline column widths */
    flex: 0 0 auto !important; /* Make toggle and button rigid by default */
}}

/* 3. Force the Title column to stretch and push others to the right */
{_HEADERS_SELECTOR} [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {{
    flex: 1 1 100px !important;
}}

/* 4. Completely strip Markdown margins and force vertical centering */
{_HEADERS_SELECTOR} [data-testid="stMarkdownContainer"] {{
    display: flex !important;
    align-items: center !important;
}}
{_HEADERS_SELECTOR} [data-testid="stMarkdownContainer"] > * {{
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
}}

/* 5. Fix Toggle internal alignment */
{_HEADERS_SELECTOR} [data-testid="stCheckbox"] {{
    display: flex !important;
    align-items: center !important;
}}
{_HEADERS_SELECTOR} [data-testid="stWidgetLabel"] {{
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}}
{_HEADERS_SELECTOR} [data-testid="stWidgetLabel"] p {{
    margin: 0 !important;
    padding: 0 !important;
    white-space: nowrap !important;
}}

/* 6. Fix Button internal alignment */
{_HEADERS_SELECTOR} [data-testid="stButton"] button {{
    margin: 0 !important;
    white-space: nowrap !important;
}}
</style>
"""

MAIN_FILTER_COUNT = 5  # Serotype, Concentration Group, Date, Sensor ID, Test ID

_SECTION_DIVIDER = (
    '<hr style="margin: 1rem 0; padding: 0; border: 0; '
    'border-top: 3px solid currentColor; opacity: 0.5;">'
)
_TITLE_TO_FILTER_DIVIDER = (
    '<hr style="margin: 0.4rem 0; padding: 0; border: 0; '
    'border-top: 1px solid currentColor; opacity: 0.25;">'
)
_FILTER_DIVIDER = (
    '<hr style="margin: 0.25rem 0; padding: 0; border: 0; '
    'border-top: 1px solid currentColor; opacity: 0.2;">'
)

FLAT_OPTIONS_THRESHOLD = 50


def _render_filter(
    label: str,
    options: list,
    default: list,
    exclude_default: bool,
    container,
    *,
    help_text: str = "",
    label_visibility: str = "collapsed",
    reset_button_key: str | None = None,
) -> tuple[list, bool]:
    """
    Render a filter: title row [Label + Exclude] ... [Reset], then selection widget.
    Returns (selected_list, exclude_bool).
    """
    use_flat = len(options) <= FLAT_OPTIONS_THRESHOLD and len(options) > 0
    if not options:
        return [], exclude_default

    # Horizontal container: Title | Exclude | [Reset]. No columns; natural flex flow.
    header = container.container(horizontal=True, key=f"filter_header_{label}")
    with header:
        st.markdown(f"### {label}")
        exclude = st.toggle(
            "Exclude",
            value=exclude_default,
            key=f"{label}_exclude",
            help="Exclude selected instead of include only.",
        )
        if reset_button_key and st.button(
            "Reset",
            key=reset_button_key,
            help="Reset selection and Exclude for this filter.",
        ):
            st.session_state[label] = []
            st.session_state[f"{label}_exclude"] = False
            st.rerun()

    if use_flat:
        selected = container.pills(
            label,
            options=options,
            default=default,
            selection_mode="multi",
            key=label,
            label_visibility=label_visibility,
        )
    else:
        selected = container.multiselect(
            label,
            options=options,
            default=default,
            help=help_text or "Leave empty to include all.",
            key=label,
            label_visibility=label_visibility,
        )
    return selected, exclude


def render_main_filter_header(container, filter_columns: list[str]) -> None:
    """
    Render the main Filters title and Reset All button in a horizontal container.
    Uses flex-wrap layout; Reset All stays rigid and wraps to next line when needed.
    """
    header = container.container(horizontal=True, key="main_filter_header")
    with header:
        st.markdown("# 🔍 Filters")
        if st.button(
            "Reset all",
            key="reset_all_filters",
            help="Reset all filter selections and Exclude toggles.",
        ):
            for col in filter_columns:
                label = format_column_label(col)
                st.session_state[label] = []
                st.session_state[f"{label}_exclude"] = False
            st.rerun()


def section_divider() -> str:
    """Return HTML for the main section divider (used after data loading)."""
    return _SECTION_DIVIDER
