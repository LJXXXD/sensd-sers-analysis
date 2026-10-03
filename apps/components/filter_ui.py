"""
Filter UI components for SERS Data Explorer.
"""

import logging

import streamlit as st

from sensd_sers_analysis.config.metadata_schema import (
    FILTER_EXPANDED_COLUMNS,
    FILTER_LONG_LABEL_AVERAGE,
    FILTER_LONG_LABEL_TOTAL,
)

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

MAIN_FILTER_COUNT = 5  # Serotype, Initial target concentration, Date, Sensor ID, Test ID

# Re-export for backwards compatibility with app.py
_FILTER_DIVIDER = FILTER_DIVIDER_HTML
_TITLE_TO_FILTER_DIVIDER = TITLE_TO_FILTER_DIVIDER_HTML


def _clear_single_filter(column: str) -> None:
    """
    Clear selection and exclude state for one canonical filter column.

    Parameters
    ----------
    column:
        Canonical dataframe column name.
    """

    clear_filter_widget_state(column)


def _use_search_selector(column: str, options: list) -> bool:
    """Choose search for a large amount of long option text.

    Parameters
    ----------
    column:
        Canonical metadata name. Core browsing dimensions stay expanded.
    options:
        Complete loaded options, before cascading filters. Character counts
        approximate text density; they do not measure browser pixel width.

    Returns
    -------
    bool
        Whether to render a searchable multiselect instead of expanded pills.
    """
    if column in FILTER_EXPANDED_COLUMNS:
        return False
    total = sum(len(str(option)) for option in options)
    return bool(options) and (
        total >= FILTER_LONG_LABEL_TOTAL and total / len(options) >= FILTER_LONG_LABEL_AVERAGE
    )


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
    layout_options: list | None = None,
) -> tuple[list, bool]:
    """
    Render a filter: title row [Label + Exclude] ... [Reset], then selection widget.
    Returns (selected_list, exclude_bool).
    """
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

    widget_args = dict(
        options=options,
        default=default,
        key=get_filter_widget_key(column),
        label_visibility=label_visibility,
        help=help_text or "Leave empty to include all.",
    )
    if _use_search_selector(column, options if layout_options is None else layout_options):
        selected = container.multiselect(label, **widget_args)
    else:
        selected = container.pills(label, selection_mode="multi", **widget_args)
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
