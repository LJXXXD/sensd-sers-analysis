"""
Filter UI components for SERS Data Explorer.
"""

import logging

import streamlit as st

from state import (
    clear_all_filter_widget_state,
    clear_filter_widget_state,
    get_filter_exclude_widget_key,
    get_filter_widget_key,
)
from theme import (
    FILTER_DIVIDER_HTML,
    SECTION_DIVIDER_HTML,
    TITLE_TO_FILTER_DIVIDER_HTML,
)

logger = logging.getLogger(__name__)

MAIN_FILTER_COUNT = 5  # Serotype, Concentration Group, Date, Sensor ID, Test ID

# Re-export for backwards compatibility with app.py
_FILTER_DIVIDER = FILTER_DIVIDER_HTML
_TITLE_TO_FILTER_DIVIDER = TITLE_TO_FILTER_DIVIDER_HTML

FLAT_OPTIONS_THRESHOLD = 50


def _clear_single_filter(column: str) -> None:
    """
    Clear selection and exclude state for one canonical filter column.

    Parameters
    ----------
    column:
        Canonical dataframe column name.
    """

    clear_filter_widget_state(column)


def _render_filter(
    column: str,
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

    header = container.container(horizontal=True, key=f"filter_header_{column}")
    with header:
        st.markdown(f"### {label}")
        exclude = st.toggle(
            "Exclude",
            value=exclude_default,
            key=get_filter_exclude_widget_key(column),
            help="Exclude selected instead of include only.",
        )
        if reset_button_key:
            st.button(
                "Reset",
                key=reset_button_key,
                help="Reset selection and Exclude for this filter.",
                on_click=_clear_single_filter,
                args=(column,),
            )

    if use_flat:
        selected = container.pills(
            label,
            options=options,
            default=default,
            selection_mode="multi",
            key=get_filter_widget_key(column),
            label_visibility=label_visibility,
        )
    else:
        selected = container.multiselect(
            label,
            options=options,
            default=default,
            help=help_text or "Leave empty to include all.",
            key=get_filter_widget_key(column),
            label_visibility=label_visibility,
        )
    return selected, exclude


def render_main_filter_header(container, filter_columns: list[str]) -> None:
    """
    Render the main Filters title and Reset All Filters button in a horizontal container.
    Uses flex-wrap layout; Reset All Filters stays rigid and wraps to next line when needed.
    """
    header = container.container(horizontal=True, key="main_filter_header")
    with header:
        st.markdown("# 🔍 Filters")
        if st.button(
            "Reset all filters",
            key="reset_all_filters",
            help="Reset all filter selections and Exclude toggles.",
        ):
            logger.info("Reset all filters for %d columns", len(filter_columns))
            clear_all_filter_widget_state(filter_columns)
            st.rerun()


def section_divider() -> str:
    """Return HTML for the main section divider (used after data loading)."""
    return SECTION_DIVIDER_HTML
