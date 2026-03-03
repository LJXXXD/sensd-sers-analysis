"""
Filter UI components for SERS Data Explorer.
"""

import streamlit as st

from sensd_sers_analysis.processing import get_filter_options
from sensd_sers_analysis.utils import format_column_label

FILTER_UI_CSS = """
<style>
/* Only filter header rows (label + exclude + reset), not number inputs */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button):not(:has([data-testid="stNumberInput"])) {
    flex-wrap: wrap;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button):not(:has([data-testid="stNumberInput"])) > div {
    white-space: nowrap;
    min-width: 0;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button):not(:has([data-testid="stNumberInput"])) > div:last-child {
    flex: 0 0 auto;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button):not(:has([data-testid="stNumberInput"])) button {
    min-width: 6rem;
}
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

    title_cols = container.columns([4, 0.6])
    with title_cols[1]:
        if reset_button_key and st.button(
            "Reset",
            key=reset_button_key,
            help="Reset selection and Exclude for this filter.",
        ):
            st.session_state[label] = []
            st.session_state[f"{label}_exclude"] = False
            st.rerun()
    with title_cols[0]:
        sub = st.columns([2, 1])
        with sub[0]:
            st.markdown(f"### {label}")
        with sub[1]:
            exclude = st.toggle(
                "Exclude",
                value=exclude_default,
                key=f"{label}_exclude",
                help="Exclude selected instead of include only.",
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


def render_filters(tidy_df, filter_columns: list[str]) -> dict:
    """
    Render the full filter UI in the sidebar and return filter_state for applying.

    Returns:
        Dict of column -> (selected_list_or_none, exclude_bool) for filtering.
    """
    st.markdown(FILTER_UI_CSS, unsafe_allow_html=True)

    filter_title_cols = st.sidebar.columns([4, 0.6])
    with filter_title_cols[0]:
        st.markdown("# 🔍 Filters")
    with filter_title_cols[1]:
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

    st.sidebar.markdown(_TITLE_TO_FILTER_DIVIDER, unsafe_allow_html=True)

    main_cols = filter_columns[:MAIN_FILTER_COUNT]
    more_cols = filter_columns[MAIN_FILTER_COUNT:]
    filter_state: dict[str, tuple[list | None, bool]] = {}

    for i, col in enumerate(main_cols):
        if i > 0:
            st.sidebar.markdown(_FILTER_DIVIDER, unsafe_allow_html=True)
        opts_all = get_filter_options(tidy_df, filter_columns, filter_state)
        help_text = "Binned concentration." if col == "concentration_group" else ""
        selected, exclude = _render_filter(
            format_column_label(col),
            opts_all[col],
            [],
            False,
            st.sidebar,
            help_text=help_text,
            reset_button_key=f"reset_{col}",
        )
        filter_state[col] = (selected if selected else None, exclude)

    with st.sidebar.expander("More Filters", expanded=False):
        for i, col in enumerate(more_cols):
            if i > 0:
                st.markdown(_FILTER_DIVIDER, unsafe_allow_html=True)
            opts_all = get_filter_options(tidy_df, filter_columns, filter_state)
            help_text = "Leave empty for no filter." if col == "filename" else ""
            selected, exclude = _render_filter(
                format_column_label(col),
                opts_all[col],
                [],
                False,
                st,
                help_text=help_text,
                reset_button_key=f"reset_more_{col}",
            )
            filter_state[col] = (selected if selected else None, exclude)

    return {k: v for k, v in filter_state.items() if k in filter_columns}


def section_divider() -> str:
    """Return HTML for the main section divider (used after data loading)."""
    return _SECTION_DIVIDER
