# SERS Data Analysis for Salmonella Detection

This toolkit is developed for the analysis of Surface-Enhanced Raman Spectroscopy (SERS) data within the [NSF SENS-D](https://sensd.ai) program. The project focuses on bridging sensor physics with computational intelligence for rapid Salmonella detection.

This specific repository hosts the research and implementation conducted by Jiahe (LJ) Li at the University of Missouri, under the supervision of Dr. Derek Anderson.

---

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

## Data inventory and research scope

The **Data inventory** tab reports all successfully loaded spectra alongside the
current sidebar-filtered subset, before sensor QA or model exclusions. Coverage
is available per recorded operator/provenance, sensor, serotype, and nominal target
concentration (CFU/mL), with acquisition dates, file basenames and composite test
sessions. CSV downloads provide the spectrum metadata ledger and coverage table. Charts
show recorded-operator/serotype counts, sensor-by-concentration coverage with a
shared color scale, and monthly acquisition counts; each chart can be downloaded
as PNG. Exact counts remain available in expandable tables.

A spectrum is one measurement, not one Raman-shift row or an independent biological
replicate. Sensor IDs do not establish unique physical devices. Sessions use the
available operator, sensor, date, test, connection and serotype metadata; missing
fields can merge sessions. Shared sensor IDs across operator/source labels and
ambiguous filename-plus-signal identities are surfaced for review. The uploader
retains basenames and embedded metadata, not original source-folder paths;
operator must not be interpreted automatically as data provider. Sparse coverage
is descriptive and never changes QA or model selection. Empty filter selections
still allow inspection of the all-loaded inventory.

- [Current research data scope and provenance](docs/DATA_SCOPE.md)
- `apps/tabs/data_inventory.py`: Streamlit presentation.
- `src/sensd_sers_analysis/application/inventory_service.py`: metadata-only counts.
- `src/sensd_sers_analysis/config/inventory.py`: grouping definitions and display defaults.

The inventory accepts existing loaded and filtered bundles without changing the
loader, preprocessing, QA, or model APIs; sample-class analyses require explicit Sample Type metadata.

## Embedded metadata and migration

The TXT-to-Excel generator provides per-signal Sample Type and Special Treatment selectors. Exported Excel contains plain values; Initial Target Concentration describes the sample before any treatment. Templates can save source-filename-keyed categorical labels. Analysis labels controls by Sample Type rather than by concentration zero.

- `src/sensd_sers_analysis/config/metadata_schema.py`: field labels and generator choices.
- `src/sensd_sers_analysis/application/metadata_presets.py`: categorical preset validation.
- `src/sensd_sers_analysis/data/metadata_migration.py`: archived-source-verified batch migration.
- [Schema, compatibility and migration details](docs/DATA_SCOPE.md#per-spectrum-metadata-schema).

The active New Format examples use this schema. Legacy files remain readable for inventory, but missing Sample Type must be completed for classification and control comparisons. Existing audit reports preserve their historical input hashes; migration records identify the current metadata changes.
