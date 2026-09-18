"""Metadata dimensions and display defaults for pre-QA data inventory."""

SPARSE_SPECTRA_DEFAULT = 3
SPECTRUM_ID_COLUMNS = ("filename", "signal_index")
SESSION_COLUMNS = ("operator", "sensor_id", "date", "test_id", "connection_id", "serotype")
COVERAGE_COLUMNS = ("sensor_id", "serotype", "target_concentration")
PROVENANCE_COLUMNS = ("source", "provider", "source_folder", "source_path", "operator")

CHART_WIDTH = 11
CHART_HEIGHT = 4
HEATMAP_ROW_HEIGHT = 0.30
HEATMAP_MIN_HEIGHT = 4
HEATMAP_PANEL_WIDTH = 5
HEATMAP_CMAP = "Blues"
CHART_COLORS = ("#28618C", "#D69A36", "#708648", "#B16B86", "#7D739F")
CHART_FONT_SIZE = 10
CHART_DPI = 160
MISSING_METADATA_LABEL = "Not recorded"

INVENTORY_GROUP_LABELS = {
    "serotype": "Serotype",
    "operator": "Operator",
    "sensor_id": "Sensor",
    "target_concentration": "Initial target concentration (CFU/mL)",
    "sample_type": "Sample type",
    "special_treatment": "Treatment",
}
SCREENING_COLORS = {"Pass": "#216B88", "Excluded": "#BF6236", "Not assessed": "#9099A0"}

INVENTORY_GROUP_LABELS["month"] = "Acquisition month"
DEFAULT_COUNT_CHARTS = ("serotype", "sensor_id", "month")
DEFAULT_COVERAGE_DIMENSIONS = ("sensor_id", "target_concentration", "serotype")
COVERAGE_FACET_COLUMNS = 3
