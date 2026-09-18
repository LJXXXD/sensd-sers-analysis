"""Data inventory tab for pre-QA coverage and acquisition provenance."""

from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sensd_sers_analysis.application.inventory_service import (
    build_data_inventory,
    build_inventory_chart_tables,
    spectrum_metadata,
)
from sensd_sers_analysis.config.inventory import CHART_DPI, SPARSE_SPECTRA_DEFAULT
from sensd_sers_analysis.visualization.inventory_plots import (
    plot_inventory_composition,
    plot_inventory_coverage,
    plot_inventory_timeline,
)


def render(loaded_wide_df: pd.DataFrame, filtered_tidy_df: pd.DataFrame) -> None:
    """Render loaded-versus-filtered coverage without running QA or a model."""
    loaded = build_data_inventory(spectrum_metadata(loaded_wide_df))
    filtered = build_data_inventory(spectrum_metadata(filtered_tidy_df, tidy=True))
    st.subheader("Data inventory")
    st.caption(
        "Pre-QA inventory of successfully loaded data. A spectrum is one measured signal, "
        "not one Raman-shift row or an independent biological replicate. Sensor IDs are "
        "labels, not confirmed unique physical devices. QA/model exclusions are not applied."
    )
    comparison = pd.concat(
        [
            loaded.summary.assign(scope="All loaded"),
            filtered.summary.assign(scope="Current filters"),
        ],
        ignore_index=True,
    ).set_index("scope")
    with st.expander("Compare all loaded data with current filters"):
        st.dataframe(comparison, hide_index=False, width="stretch")
    scope = st.radio(
        "Inventory details",
        ["Current filters", "All loaded"],
        horizontal=True,
        key="inventory_scope",
    )
    inventory = filtered if scope == "Current filters" else loaded
    summary = inventory.summary.iloc[0]
    for column, label, field in zip(
        st.columns(4),
        ("Measured spectra", "Sensor IDs", "File basenames", "Sessions"),
        ("spectra", "sensor_ids", "file_basenames", "sessions"),
    ):
        column.metric(label, int(summary[field]))
    if pd.notna(summary["first_date"]):
        st.caption(
            f"Acquired {summary['first_date']:%Y-%m-%d} to "
            f"{summary['last_date']:%Y-%m-%d} · {scope}"
        )
    chart_tables = build_inventory_chart_tables(inventory.spectra)
    if inventory.spectra.empty:
        st.info("No spectra in this scope. Select All loaded or adjust the sidebar filters.")
    else:
        for name, plot in (
            ("composition", plot_inventory_composition),
            ("coverage", plot_inventory_coverage),
            ("timeline", plot_inventory_timeline),
        ):
            if chart_tables[name].empty:
                continue
            figure = plot(chart_tables[name])
            st.pyplot(figure, width="stretch")
            image = BytesIO()
            figure.savefig(image, format="png", dpi=CHART_DPI, bbox_inches="tight")
            st.download_button(
                "Download chart PNG",
                image.getvalue(),
                f"data_inventory_{name}.png",
                mime="image/png",
                key=f"inventory_chart_{name}",
            )
            plt.close(figure)
            if name == "coverage":
                st.caption(
                    "Each cell counts measured spectra. A dash means none in the selected "
                    "data, not a failed sensor or proof that a measurement was never made. "
                    "Facets use the same count scale and sensor order."
                )
            if name == "timeline":
                st.caption(
                    f"Monthly counts exclude {int(summary['missing_or_invalid_dates'])} "
                    "spectra with missing or invalid acquisition dates. "
                    "Empty months indicate no dated records in this scope."
                )
    st.caption(
        "Dates are acquisition metadata, not receipt dates or a claim that all data through "
        "that date were received. File counts are distinct basenames. Sessions are observed "
        "operator × sensor × date × test × connection × serotype combinations; missing "
        "metadata can merge sessions. T1 alone is not a session identifier."
    )
    if loaded.duplicate_id_rows:
        st.warning(
            f"{loaded.duplicate_id_rows} loaded rows share filename + signal_index identities. "
            "Repeated uploads or colliding basenames may make counts ambiguous; verify filenames."
        )
    st.markdown("#### Definitions and exact counts")
    st.caption(
        "Operator is embedded acquisition metadata, not necessarily the person who provided "
        "the file. The uploader does not retain the original folder path. Source/provider "
        "fields appear only when present in metadata; they are not inferred from operator names."
    )
    with st.expander("Source and recorded operator counts"):
        st.dataframe(inventory.provenance, hide_index=True, width="stretch")
    if not inventory.shared_ids.empty:
        st.write(
            "Sensor IDs shared across provenance/operator labels (physical identity unverified)"
        )
        st.dataframe(inventory.shared_ids, hide_index=True, width="stretch")
    with st.expander("Per-sensor counts"):
        st.dataframe(inventory.sensors, hide_index=True, width="stretch")
    st.caption(
        "Initial target concentration is the nominal CFU/mL before any special treatment. Only observed "
        "combinations are listed; absent combinations are not fabricated as zero-count groups."
    )
    with st.expander("Sensor × serotype × target concentration counts"):
        st.dataframe(inventory.coverage, hide_index=True, width="stretch")
    threshold = st.number_input(
        "Highlight groups with at most this many spectra",
        min_value=1,
        value=SPARSE_SPECTRA_DEFAULT,
        step=1,
        key="inventory_sparse_threshold",
        help="Descriptive coverage threshold only; it never excludes data or changes a model.",
    )
    if not inventory.coverage.empty:
        sparse = inventory.coverage.loc[inventory.coverage["spectra"] <= threshold]
        st.write(f"{len(sparse)} observed groups have at most {threshold} spectra.")
        with st.expander("Inspect sparse observed groups"):
            st.dataframe(sparse, hide_index=True, width="stretch")
    with st.expander("Missing metadata and spectrum ledger"):
        st.dataframe(inventory.missing_metadata, hide_index=True, width="stretch")
        st.dataframe(inventory.spectra, hide_index=True, width="stretch")
    for label, table, filename in (
        ("Download coverage CSV", inventory.coverage, "data_inventory_coverage.csv"),
        ("Download spectrum ledger CSV", inventory.spectra, "data_inventory_spectra.csv"),
    ):
        st.download_button(
            label,
            table.to_csv(index=False).encode("utf-8-sig"),
            filename,
            mime="text/csv",
            key=f"inventory_{filename}",
        )
