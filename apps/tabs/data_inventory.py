"""Curated inventory views with optional chart customization."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sensd_sers_analysis.application.inventory_service import (
    build_data_inventory,
    build_inventory_counts,
    spectrum_metadata,
)
from sensd_sers_analysis.config.inventory import (
    DEFAULT_COUNT_CHARTS,
    DEFAULT_COVERAGE_DIMENSIONS,
    INVENTORY_GROUP_LABELS,
)
from sensd_sers_analysis.visualization.inventory_plots import (
    plot_inventory_composition,
    plot_inventory_coverage,
)


def _add_chart(kind: str) -> None:
    """Append a stable chart identity without resetting existing chart choices."""
    ids = st.session_state[f"inventory_{kind}_ids"]
    next_id = st.session_state.get(f"inventory_{kind}_next", max(ids, default=-1) + 1)
    st.session_state[f"inventory_{kind}_ids"] = [*ids, next_id]
    st.session_state[f"inventory_{kind}_next"] = next_id + 1


def _remove_chart(kind: str, chart_id: int) -> None:
    """Remove one chart while preserving the identities of all remaining charts."""
    st.session_state[f"inventory_{kind}_ids"] = [
        value for value in st.session_state[f"inventory_{kind}_ids"] if value != chart_id
    ]
    for suffix in ("group", "color", "row", "column", "facet"):
        st.session_state.pop(f"inventory_{kind}_{chart_id}_{suffix}", None)


def render(filtered_tidy_df: pd.DataFrame) -> None:
    """Show useful default views, leaving dimension controls available for exploration."""
    inventory = build_data_inventory(spectrum_metadata(filtered_tidy_df, tidy=True))
    summary = inventory.summary.iloc[0]
    st.subheader("Data inventory")
    for column, label, field in zip(
        st.columns(3), ("Spectra", "Sensors", "Files"), ("spectra", "sensor_ids", "file_basenames")
    ):
        column.metric(label, int(summary[field]))
    if pd.notna(summary["first_date"]):
        st.caption(f"{summary['first_date']:%Y-%m-%d} – {summary['last_date']:%Y-%m-%d}")
    if inventory.spectra.empty:
        st.info("No data matches the sidebar filters.")
        return
    count_ids = st.session_state.setdefault(
        "inventory_count_ids", list(range(len(DEFAULT_COUNT_CHARTS)))
    )
    for chart_id in count_ids:
        key = f"inventory_count_{chart_id}_group"
        if key not in st.session_state:
            used = {
                st.session_state.get(f"inventory_count_{other}_group")
                for other in count_ids
                if other != chart_id
            }
            preferred = list(DEFAULT_COUNT_CHARTS) + list(INVENTORY_GROUP_LABELS)
            st.session_state[key] = next(
                (value for value in preferred if value not in used), preferred[0]
            )
        control, color_control, remove = st.columns([6, 6, 1])
        group = control.selectbox(
            "Group by",
            list(INVENTORY_GROUP_LABELS),
            key=key,
            format_func=INVENTORY_GROUP_LABELS.get,
        )
        color_key = f"inventory_count_{chart_id}_color"
        color_options = [None] + [value for value in INVENTORY_GROUP_LABELS if value != group]
        if color_key not in st.session_state or st.session_state[color_key] not in color_options:
            st.session_state[color_key] = (
                "target_concentration" if group == "serotype" else "serotype"
            )
        segment = color_control.selectbox(
            "Color by",
            color_options,
            key=color_key,
            format_func=lambda value: "None" if value is None else INVENTORY_GROUP_LABELS[value],
        )
        remove.button(
            "−",
            key=f"remove_count_{chart_id}",
            help="Remove this chart",
            on_click=_remove_chart,
            args=("count", chart_id),
        )
        counts = build_inventory_counts(
            inventory.spectra, (group, segment) if segment else (group,)
        )
        figure = plot_inventory_composition(counts, group_by=group, color_by=segment)
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    st.button(
        "+ Add count chart",
        on_click=_add_chart,
        args=("count",),
    )
    st.subheader("Counts across three dimensions")
    coverage_ids = st.session_state.setdefault("inventory_coverage_ids", [0])
    for chart_id in coverage_ids:
        with st.expander("Choose rows, columns and panels"):
            selected = []
            for suffix, label, default in zip(
                ("row", "column", "facet"),
                ("Rows", "Columns", "Separate panels"),
                DEFAULT_COVERAGE_DIMENSIONS,
            ):
                key = f"inventory_coverage_{chart_id}_{suffix}"
                options = [value for value in INVENTORY_GROUP_LABELS if value not in selected]
                if st.session_state.get(key) not in options:
                    st.session_state[key] = default if default in options else options[0]
                selected.append(
                    st.selectbox(label, options, key=key, format_func=INVENTORY_GROUP_LABELS.get)
                )
            st.button(
                "− Remove coverage chart",
                key=f"remove_coverage_{chart_id}",
                on_click=_remove_chart,
                args=("coverage", chart_id),
            )
        counts = build_inventory_counts(inventory.spectra, tuple(selected))
        figure = plot_inventory_coverage(
            counts,
            row_dimension=selected[0],
            column_dimension=selected[1],
            facet_dimension=selected[2],
        )
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    st.button("+ Add coverage chart", on_click=_add_chart, args=("coverage",))
