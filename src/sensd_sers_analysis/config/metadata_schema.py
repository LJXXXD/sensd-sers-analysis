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
