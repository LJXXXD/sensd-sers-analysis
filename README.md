# SERS Data Analysis for Salmonella Detection

This toolkit is developed for the analysis of Surface-Enhanced Raman Spectroscopy (SERS) data within the [NSF SENS-D](https://sensd.ai) program. The project focuses on bridging sensor physics with computational intelligence for rapid Salmonella detection.

This specific repository hosts the research and implementation conducted by Jiahe (LJ) Li at the University of Missouri, under the supervision of Dr. Derek Anderson.

---

## Default dataset

The analysis app starts with the bundled Adheesha 2025/2026 Dilutions and Amjad New Format workbooks in `example_data/`. Heat kill, PAA and Repeatability collections are not loaded by default. All dates and serotypes are included before sidebar filtering. Uploading files replaces the bundled selection. **Unload data** keeps the session empty across reruns; **Load example data** restores the bundled collection. Both sources use the same parsing pipeline. Collection paths are configured in `src/sensd_sers_analysis/config/example_data.py`.

## Installation

This project is packaged via standard `pyproject.toml` and requires Python 3.12+.

### Developers

Clone the repo and install in editable mode with dev dependencies (testing, linting, notebooks):

```bash
# Clone
git clone https://github.com/LJXXXD/sensd-sers-analysis.git
cd sensd-sers-analysis

# Via uv (recommended)
uv sync
uv run pre-commit install   # optional
uv run streamlit run apps/app.py   # run app

# Or via pip
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -e . --group dev
pre-commit install   # optional
streamlit run apps/app.py   # run app
```

---

## Contact

**Jiahe (LJ) Li** — j.li@missouri.edu — University of Missouri

## Integration-time normalization

The App enables integration-time normalization by default: `I_reference = I_input × 100 ms / integration_time_ms`. The sidebar toggle restores original intensities when disabled. Scaling precedes shared spectral alignment and feature extraction, so spectra, peaks, QA and model inputs use the same representation. Excel files and acquisition metadata are unchanged. Invalid exposure values block enabled normalization and are listed for review. The calculation assumes approximately linear unsaturated response and input values not already exposure-normalized; scan averages are not treated as summed exposure counts. Historical results require rerunning before comparison with normalized results. The core `build_derived_bundle` API retains an explicit opt-in `normalize_exposure` argument for existing scripts.

## App organization

The first three pages follow Data → Features → Sensor quality. Data contains Inventory and Spectra; Features contains peak discovery, fixed-anchor extraction and feature analysis; Sensor quality contains Screening and Variability diagnostics. Models groups classification and the three regression approaches; Validation remains a separate page. Basic integrated-intensity features are computed by the shared pipeline before screening; the page order communicates the workflow rather than triggering preprocessing.

## Data inventory and research scope

The **Data inventory** tab follows the sidebar filters and summarizes spectra, sensors and files. Three count charts default to serotype (colored by initial target concentration), sensor (colored by serotype), and acquisition month (colored by serotype). Each chart exposes separate Group by and Color by selectors; Color by None shows a single-color total. Charts can be added or removed; new charts prefer an unused primary dimension, while repeated primary dimensions remain available for different color comparisons. Coverage defaults to sensor rows, concentration columns and serotype panels; charts can be added or removed, with three distinct dimensions configurable in a collapsed control panel. The full metadata ledger is not displayed here. Spectra Viewer uses compact color/display selectors, with line style and height under More plot options.

The **Sensor assessment** tab starts with per-sensor screening counts using the configured classification QA feature. Counts include controls belonging to each sensor/serotype pair and precede final spectrum cleaning; no deduplication is applied. Insufficient or non-finite fits appear as Not assessed. Selecting a pair displays the existing cleaned regression, optional same-serotype passing-sensor comparison on shared axes, a fit-point ledger and its spectra. Fit outliers are distinct from whole-pair exclusions and from final model preparation. Advanced regression diagnostics remain available on demand.

Data Loading reports unreadable files and per-spectrum metadata issues with filenames and field values. Acquisition dates accept mixed date-only and timestamp formats. Sensor IDs represent devices in the UI; provenance conflicts still require source-data review. All charts describe the current data selection, not independent biological replicate counts or validated device yield.

- [Current research data scope and provenance](docs/DATA_SCOPE.md)
- `apps/tabs/data_inventory.py`: inventory presentation.
- `apps/components/screening_summary.py`: screening explanation and drill-down.
- `src/sensd_sers_analysis/application/inventory_service.py`: metadata counts and diagnostics.
- `src/sensd_sers_analysis/application/sensor_assessment_service.py`: screening counts and fit-point provenance.
- `src/sensd_sers_analysis/config/inventory.py`: grouping definitions and display defaults.

## Embedded metadata and migration

The TXT-to-Excel generator provides per-signal Sample Type and Special Treatment selectors. Exported Excel contains plain values; Initial Target Concentration describes the sample before any treatment. Templates can save source-filename-keyed categorical labels. Analysis labels controls by Sample Type rather than by concentration zero.

- `src/sensd_sers_analysis/config/metadata_schema.py`: field labels and generator choices.
- `src/sensd_sers_analysis/application/metadata_presets.py`: categorical preset validation.
- `src/sensd_sers_analysis/data/metadata_migration.py`: archived-source-verified batch migration.
- [Schema, compatibility and migration details](docs/DATA_SCOPE.md#per-spectrum-metadata-schema).

The active New Format examples use this schema. Legacy files remain readable for inventory, but missing Sample Type must be completed for classification and control comparisons. Existing audit reports preserve their historical input hashes; migration records identify the current metadata changes.

## Local research records

`reports/` contains local inventory/migration evidence, model experiments and report deliverables. The directory is ignored by Git and excluded from HF deployment. Interpret each record with its documented input hashes, runtime, split protocol and historical scope; it does not describe current App results.

Local archives preserve scripts, numerical arrays, predictions, telemetry and manifest-listed logs; use a separate copy for reruns. Presentation previews/inspection files live in ignored local support storage. The four tracked raw TXT inputs in `example_data/txt_to_excel/` are manual converter examples and retain instrument-export bytes independently of the default workbook loader.
