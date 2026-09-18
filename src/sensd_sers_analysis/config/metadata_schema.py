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
    "concentration_group": "Actual concentration (CFU/mL)",
    "special_treatment": "Treatment",
}
