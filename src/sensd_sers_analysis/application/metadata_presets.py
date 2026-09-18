"""Validation of filename-keyed categorical metadata presets."""

from sensd_sers_analysis.config.metadata_schema import (
    SAMPLE_TYPE_OPTIONS,
    SPECIAL_TREATMENT_OPTIONS,
)


def validate_signal_preset(payload: object) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Validate categorical labels without guessing or mapping by file position."""
    if not isinstance(payload, dict):
        return {}, ["Invalid signal_labels: expected a filename-keyed object."]
    valid = {}
    warnings = []
    for filename, labels in payload.items():
        if not isinstance(filename, str) or not isinstance(labels, dict):
            warnings.append("Invalid per-signal preset entry ignored.")
            continue
        sample_type = labels.get("sample_type")
        treatment = labels.get("special_treatment", "None")
        if sample_type not in SAMPLE_TYPE_OPTIONS or treatment not in SPECIAL_TREATMENT_OPTIONS:
            warnings.append(f"Unsupported sample type or treatment for {filename}; ignored.")
            continue
        valid[filename] = {"sample_type": sample_type, "special_treatment": treatment}
    return valid, warnings
