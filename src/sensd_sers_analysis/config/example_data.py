"""Repository-relative bundled dilution collections for the analysis app."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DATA_DIRECTORIES = (
    PROJECT_ROOT / "example_data/Adheesha - New Format/_Data for ML-New-2025/Dilutions",
    PROJECT_ROOT / "example_data/Adheesha - New Format/_Data for ML New-2026/Dilutions",
    PROJECT_ROOT / "example_data/Amjad - New Format",
)
