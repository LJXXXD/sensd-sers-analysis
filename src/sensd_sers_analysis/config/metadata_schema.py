"""Canonical per-spectrum metadata labels and generator choices."""

SAMPLE_TYPE_LABEL = "Sample Type"
SAMPLE_TYPE_OPTIONS = ("Rinsate control", "Bacteria sample")
SPECIAL_TREATMENT_OPTIONS = ("None", "Heat treated", "PAA 10 ppm", "PAA 50 ppm", "PAA 100 ppm")
INITIAL_TARGET_LABEL = "Initial Target Concentration (CFU/mL)"
PER_SIGNAL_PRESET_KEY = "_txt2excel_signal_preset"

METADATA_DISPLAY_LABELS = {
    "disk_diameter_nm": "Disk diameter (nm)",
    "periodicity_um": "Periodicity (µm)",
    "thickness_nm": "Thickness (nm)",
    "core_diameter_um": "Core diameter (µm)",
    "integration_time_ms": "Integration time (ms)",
    "scan_average": "Scans averaged",
    "target_concentration": "Initial target concentration (CFU/mL)",
    "target_concentration_group": "Initial target concentration (CFU/mL)",
    "concentration": "Actual concentration (CFU/mL)",
    "concentration_group": "Initial target concentration (CFU/mL)",
    "special_treatment": "Treatment",
}

SPECTRA_HIDDEN_FACTORS = ("concentration_group", "target_concentration_group", "log_concentration")
SPECTRA_CONTINUOUS_FACTORS = ("concentration",)

# Sidebar presentation uses the complete loaded option set, not cascading subsets.
FILTER_EXPANDED_COLUMNS = frozenset(
    {"serotype", "target_concentration", "date", "sensor_id", "test_id"}
)
FILTER_LONG_LABEL_AVERAGE = 24
FILTER_LONG_LABEL_TOTAL = 320


SPECTRA_FACTOR_ORDER = (
    "serotype",
    "target_concentration",
    "concentration",
    "sample_type",
    "special_treatment",
    "sensor_id",
    "date",
    "test_id",
    "connection_id",
    "operator",
    "integration_time_ms",
    "scan_average",
    "sensor_model",
    "disk_diameter_nm",
    "periodicity_um",
    "thickness_nm",
    "core_diameter_um",
    "rinsate_type",
    "testing_time",
    "filename",
    "source_txt_filename",
    "signal_index",
)


# Workbook field order and JSON preset keys are shared serialization contracts.
NUMERIC_METADATA_FIELD_NUMBERS = frozenset({1, 2, 3, 4, 6, 7})


DATE_METADATA_FIELD_NUMBERS = frozenset({13})


TIME_METADATA_FIELD_NUMBERS = frozenset({14})


OPTIONAL_METADATA_FIELD_NUMBERS = frozenset({16})


METADATA_FIELD_SPECS: tuple[tuple[int, str, str], ...] = (
    (1, "Disk Diameter (nm)", "txt2excel_meta_disk_diameter_nm"),
    (2, "Periodicity (µm)", "txt2excel_meta_periodicity_um"),
    (3, "Thickness (nm)", "txt2excel_meta_thickness_nm"),
    (4, "Core Diameter (µm)", "txt2excel_meta_core_diameter_um"),
    (5, "Sensor Model", "txt2excel_meta_sensor_model"),
    (6, "Integration Time (ms)", "txt2excel_meta_integration_time_ms"),
    (7, "Scan Average", "txt2excel_meta_scan_average"),
    (8, "Sensor ID", "txt2excel_meta_sensor_id"),
    (9, "Test ID", "txt2excel_meta_test_id"),
    (10, "Connection ID", "txt2excel_meta_connection_id"),
    (11, "Serotype", "txt2excel_meta_serotype"),
    (12, "Rinsate Type", "txt2excel_meta_rinsate_type"),
    (13, "Date", "txt2excel_meta_date"),
    (14, "Testing Time", "txt2excel_meta_testing_time"),
    (15, "Operator", "txt2excel_meta_operator"),
    (16, "Notes", "txt2excel_meta_notes"),
)


METADATA_LOGICAL_GROUPS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4, 5),
    (6, 7),
    (8, 9, 10),
    (11, 12),
    (13, 14, 15),
    (16,),
)


METADATA_WIDGET_KEYS = frozenset(widget_key for _, _, widget_key in METADATA_FIELD_SPECS)
