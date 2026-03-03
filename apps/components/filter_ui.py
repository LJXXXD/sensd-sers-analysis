"""
Filter UI components for SERS Data Explorer.
"""

import streamlit as st

from sensd_sers_analysis.utils import format_column_label

from theme import (
    FILTER_DIVIDER_HTML,
    SECTION_DIVIDER_HTML,
    TITLE_TO_FILTER_DIVIDER_HTML,
)

MAIN_FILTER_COUNT = 5  # Serotype, Concentration Group, Date, Sensor ID, Test ID

# Re-export for backwards compatibility with app.py
_FILTER_DIVIDER = FILTER_DIVIDER_HTML
_TITLE_TO_FILTER_DIVIDER = TITLE_TO_FILTER_DIVIDER_HTML

FLAT_OPTIONS_THRESHOLD = 50


def _clear_single_filter(lbl: str) -> None:
    """Clear selection and exclude for a single filter (used by Reset button)."""
    st.session_state[lbl] = []
    st.session_state[f"{lbl}_exclude"] = False


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

    header = container.container(horizontal=True, key=f"filter_header_{label}")
    with header:
        st.markdown(f"### {label}")
        exclude = st.toggle(
            "Exclude",
            value=exclude_default,
            key=f"{label}_exclude",
            help="Exclude selected instead of include only.",
        )
        if reset_button_key:
            st.button(
                "Reset",
                key=reset_button_key,
                help="Reset selection and Exclude for this filter.",
                on_click=_clear_single_filter,
                args=(label,),
            )

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
            for col in filter_columns:
                label = format_column_label(col)
                st.session_state[label] = []
                st.session_state[f"{label}_exclude"] = False
            st.rerun()


def section_divider() -> str:
    """Return HTML for the main section divider (used after data loading)."""
    return SECTION_DIVIDER_HTML
