"""Synthetic inventory tests; no experimental data or model execution."""

import pandas as pd
import pytest

from sensd_sers_analysis.application.inventory_service import (
    build_data_inventory,
    spectrum_metadata,
)


@pytest.fixture
def spectra():
    """Create shared IDs, reused T1 labels, missing metadata and sparse groups."""
    return pd.DataFrame(
        {
            "filename": ["a.xlsx", "a.xlsx", "b.xlsx", "c.xlsx"],
            "signal_index": [0, 1, 0, 0],
            "sensor_id": ["AM1", "AM1", "AM1", "S2"],
            "operator": ["Adheesha", "Adheesha", "Amjad", ""],
            "serotype": ["ST", "ST", "ST", "SE"],
            "date": ["2025-08-01", "2025-08-01", "2025-09-01", "invalid"],
            "test_id": ["T1"] * 4,
            "connection_id": ["C1"] * 4,
            "target_concentration": [10, 10, 100, 0],
            "concentration": [8, 12, 95, 0],
            "rs_500": [1.0] * 4,
        }
    )


def test_inventory_counts_spectra_and_composite_sessions(spectra):
    """Count signals rather than signal pixels or globally reused test labels."""
    inventory = build_data_inventory(spectrum_metadata(spectra))
    summary = inventory.summary.iloc[0]
    assert summary["spectra"] == 4
    assert summary["sensor_ids"] == 2
    assert summary["sessions"] == 3
    assert summary["file_basenames"] == 3
    assert summary["missing_or_invalid_dates"] == 1
    assert summary["first_date"] == pd.Timestamp("2025-08-01")
    assert inventory.coverage["spectra"].sum() == 4
    assert sorted(inventory.coverage["spectra"].tolist()) == [1, 1, 2]
    assert inventory.shared_ids.iloc[0]["sensor_id"] == "AM1"
    assert inventory.shared_ids.iloc[0]["distinct_operator"] == 2
    assert "rs_500" not in inventory.spectra
    assert inventory.missing_metadata.set_index("field").loc["operator", "missing_spectra"] == 1


def test_filtered_tidy_counts_one_record_per_spectrum(spectra):
    """Repeated Raman-shift rows do not inflate counts, and filters stay separate."""
    metadata = spectrum_metadata(spectra.iloc[:2])
    tidy = pd.concat([metadata.assign(raman_shift=500), metadata.assign(raman_shift=600)])
    tidy["intensity"] = 1.0
    inventory = build_data_inventory(spectrum_metadata(tidy, tidy=True))
    assert inventory.summary.iloc[0]["spectra"] == 2
    assert inventory.summary.iloc[0]["sessions"] == 1
    assert len(spectra) == 4


def test_duplicate_identities_are_retained_and_flagged(spectra):
    """Do not silently drop repeated wide rows or claim physical device identity."""
    repeated = pd.concat([spectra, spectra.iloc[:1]], ignore_index=True)
    inventory = build_data_inventory(spectrum_metadata(repeated))
    assert inventory.summary.iloc[0]["spectra"] == 5
    assert inventory.duplicate_id_rows == 2


@pytest.mark.parametrize("records", [pd.DataFrame(), pd.DataFrame({"sensor_id": ["S1"]})])
def test_empty_and_partial_metadata(records):
    """Unavailable date or identity fields remain safe and explicitly missing."""
    inventory = build_data_inventory(records)
    assert inventory.summary.iloc[0]["spectra"] == len(records)
    assert inventory.summary.iloc[0]["missing_or_invalid_dates"] == len(records)
    assert inventory.coverage.empty or inventory.coverage["spectra"].sum() == len(records)


def test_chart_tables_conserve_counts_and_preserve_missing_labels(spectra):
    """Charts retain missing metadata and omit only undated records from timeline."""
    from sensd_sers_analysis.application.inventory_service import build_inventory_chart_tables

    tables = build_inventory_chart_tables(spectra)
    assert tables["composition"]["spectra"].sum() == 4
    assert tables["coverage"]["spectra"].sum() == 4
    assert tables["timeline"]["spectra"].sum() == 3
    assert "Not recorded" in tables["composition"]["operator"].tolist()
    assert tables["timeline"]["month"].tolist() == ["2025-08", "2025-09"]


def test_timeline_includes_internal_empty_calendar_months(spectra):
    """Zero bars refer to missing records in this scope, not a truncated date axis."""
    from sensd_sers_analysis.application.inventory_service import build_inventory_chart_tables

    spectra["date"] = ["2025-07-01", "2025-07-01", "2025-09-01", "invalid"]
    tables = build_inventory_chart_tables(spectra)
    assert tables["timeline"]["spectra"].tolist() == [2, 0, 1]


def test_inventory_charts_render_with_exact_heatmap_total(spectra):
    """All plotted heatmap cells reconcile with spectrum counts."""
    import matplotlib.pyplot as plt
    import numpy as np

    from sensd_sers_analysis.application.inventory_service import build_inventory_chart_tables
    from sensd_sers_analysis.visualization.inventory_plots import (
        plot_inventory_composition,
        plot_inventory_coverage,
        plot_inventory_timeline,
    )

    tables = build_inventory_chart_tables(spectra)
    figures = [
        plot_inventory_composition(tables["composition"]),
        plot_inventory_coverage(tables["coverage"]),
        plot_inventory_timeline(tables["timeline"]),
    ]
    for figure in figures:
        figure.canvas.draw()
    heatmap_axes = figures[1].axes[:2]
    assert np.isclose(sum(axis.collections[0].get_array().sum() for axis in heatmap_axes), 4)
    for figure in figures:
        plt.close(figure)


def test_serotype_composition_uses_concentration_segments(spectra):
    """Serotype bars retain concentration coverage rather than collapsing to a single color."""
    from sensd_sers_analysis.application.inventory_service import build_inventory_chart_tables

    table = build_inventory_chart_tables(spectra, group_by="serotype")["composition"]
    assert {"serotype", "target_concentration", "spectra"}.issubset(table.columns)
    assert table.spectra.sum() == len(spectra)
    assert set(table.target_concentration) == {"0", "10", "100"}


def test_month_composition_preserves_unknown_dates(spectra):
    """Missing dates remain visible as an unknown group in the configurable count chart."""
    from sensd_sers_analysis.application.inventory_service import build_inventory_counts

    table = build_inventory_counts(spectra, ("month", "serotype"))
    assert table.spectra.sum() == len(spectra)
    assert "Not recorded" in table.month.tolist()


def test_custom_coverage_conserves_counts(spectra):
    """Alternative facet/row/column assignments retain every spectrum."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sensd_sers_analysis.application.inventory_service import build_inventory_counts
    from sensd_sers_analysis.visualization.inventory_plots import plot_inventory_coverage

    dims = ("operator", "serotype", "target_concentration")
    table = build_inventory_counts(spectra, dims)
    fig = plot_inventory_coverage(
        table, row_dimension=dims[0], column_dimension=dims[1], facet_dimension=dims[2]
    )
    assert np.isclose(
        sum(axis.collections[0].get_array().sum() for axis in fig.axes[:-1]), len(spectra)
    )
    plt.close(fig)
