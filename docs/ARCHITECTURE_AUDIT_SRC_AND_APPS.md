# Architecture and maintenance review

This document describes the current `src/sensd_sers_analysis` and `apps` implementation. It records responsibilities and maintenance questions, not permanent directory requirements or authorization for proposed migrations. Scientific contracts and algorithms are described in [THEORY_AND_IMPLEMENTATION.md](THEORY_AND_IMPLEMENTATION.md); input scope and metadata are described in [DATA_SCOPE.md](DATA_SCOPE.md).

## Current responsibilities

| Area | Responsibility |
| --- | --- |
| `apps/app.py` | Streamlit composition, analysis/prep mode selection, sidebar controls, and tab dispatch |
| `apps/components`, `apps/tabs` | Widgets, messages, tables, figures, and downloads |
| `apps/cache.py`, `apps/state.py` | Streamlit caching and session-state adapters |
| `application` | Dataset preparation, filtering, inventory, assessment/model orchestration, and artifact contracts |
| `data`, `processing` | Embedded-workbook parsing, metadata, normalization, grid alignment, features, and filters |
| `assessment`, `classification`, `regression` | Scientific calculations, screening, model preparation, training, and evaluation |
| `visualization`, `report` | Figure construction and PDF assembly |
| `config`, `utils` | Shared settings/policies and small utilities with identifiable ownership |

The library currently has no dependency on Streamlit or `apps`. Streamlit cache decorators live in `apps/cache.py` and the upload component; the application services remain callable independently of the UI.

`apps/txt_to_excel.py` owns prep widgets, session state, metadata forms and template actions. `data/txt_converter.py` owns callable TXT parsing, metadata serialization, preset conversion and workbook construction. `config/metadata_schema.py` owns field order, field types and canonical preset keys. The library has no Streamlit dependency.

## Active analysis flow

1. `render_data_source` chooses bundled dilution workbooks or uploaded bytes. Uploads replace the bundled selection; Unload retains an empty state and Load example data restores bundled input.
2. `load_uploaded_bundle` parses embedded workbooks into wide/tidy data and a per-file load report. File names and bytes participate in the upload cache key.
3. `build_derived_bundle` enriches metadata, optionally normalizes integration time, aligns native Raman grids, trims the spectral window, and builds basic/PCA/dynamic-peak artifacts.
4. Filter services produce aligned tidy and feature views. Inventory and Spectra use the filtered pre-QA data; targeted peak features are merged for downstream analysis.
5. Tabs call cached services or domain functions and render their results. PDF actions assemble reports from the corresponding data/artifacts.

The default bundled source uses the three directories in `config/example_data.py`, not every file under `example_data`. Raw TXT inputs under `example_data/txt_to_excel` are available for manual prep-mode use and are not part of the bundled workbook loader.

## Current navigation

| Main tab | Views and modules |
| --- | --- |
| Data | Inventory (`data_inventory`), Spectra (`spectra_viewer`) |
| Features | Discover peaks (`peak_discovery`), Extract features (`peak_feature_extraction`), Analyze features (`feature_analysis`) |
| Sensor quality | Screening (`sensor_assessment`), Variability diagnostics (`sensor_qc`) |
| Models | Classification (`serotype_classification`), Global/Two-Stage/MTL regression |
| Validation | Validation tables (`validation_metrics`) |

These are nested tabs in the current UI. Further navigation integration remains a product-design question; the table does not establish the desired final layout.

## Numerical and artifact boundaries

- Basic feature integration uses the actual Raman coordinates. Non-finite integrals or fewer than two coordinates produce an unavailable area rather than a sum of intensity samples.
- Uploaded filenames are simple, unique basenames; case/Unicode-equivalent names fail before writing. The existing `(filename, signal_index)` spectrum identity does not support duplicate uploaded basenames. All non-spectral metadata survives wide-to-tidy conversion; reserved computed/spectral names and duplicate normalized metadata fields produce explicit per-file failures.
- Workbook coordinates and intensities must be finite, with distinct Raman labels at the loader's two-decimal precision. Rows containing intensity data within the parsed block require a coordinate; only fully blank separator rows may be ignored. Invalid files appear in the load report instead of supplying ambiguous or substituted measurements.
- Classification and concentration preparation retain original row labels and source order. Regression targets require finite log values and finite positive actual CFU. Validation maps predictions by unique source labels while split arrays/masks remain positional. Failed model fits abort that Validation request with a visible error; unavailable sensor holdout has no fallback fit and no ML scores.
- MTL's seed controls CPU PyTorch initialization, dropout and shuffling as well as its NumPy sub-split. Its scoped RNG ownership restores the caller's CPU state; runtime/hardware differences remain outside the reproducibility guarantee.
- `MacroRegressionResult` owns pooled coordinates, sensor labels, fit coefficients, metrics, and the point-aligned outlier mask. Macro plotting consumes that artifact directly; without a supplied result it computes one once.
- Other assessment plotting functions can still call fitting helpers. Passing existing results is useful where it eliminates repeated work; splitting every plotting helper into a new service is not inherently necessary.
- Cohort PCA and dynamic peak discovery are exploratory. PCA uses complete finite spectra; dynamic anchor means use complete bacteria spectra. Neither supplies default model predictors. Selected scalar/fixed-peak predictors must be finite; missing values are never substituted with measured zeros.
- Model eligibility uses explicit sample identity and actual positive CFU for regression, independently of retrospective QA. Classifiers and regressors hold out sensor IDs. Fold-local scaling belongs to estimator Pipelines; candidate families share a realized training-sensor CV plan and scorer. Training CV selects the family, with a fixed RF reference when CV is unavailable. No outer test metric selects a family.
- MTL uses a group-disjoint inner early-stopping split, inner-training scaling and training-only class vocabulary. Unsupported classes or invalid losses abort. Restored best weights use a fresh optimizer and a fixed full-training fine-tuning budget.
- Validation classifier and regression use the same held-in sensor assignment. Every requested fold must succeed; unsupported class support or model failure aborts the request. Missing sensor holdout produces no fabricated scores.
- Figure factories return caller-owned standalone Figures. Report services register only their own figures in an ExitStack; failures release those figures without touching unrelated callers. Typed `PlotUnavailableError` describes expected absent plot data and appears as an explicit report note; other plotting failures abort export.
- PDF reports preserve every table row/header and all metric columns through wrapped cells and horizontal panels. Repeated row identity disambiguates panels when the first column is nonunique. Figures retain aspect ratio. Assessment and degradation scopes are explicit, and missing sections contain availability explanations.
- PDF and converter download bytes are bound to the current data and settings. Scope changes or failed requests clear stale bytes; generation exposes a download only after success.

## Maintenance boundaries

| Area | Current ownership and boundary |
| --- | --- |
| Model evaluation | Shared `modeling.py` owns strict feature matrices, Pipeline fitting/search and comparable CV family selection; `splits.py` owns sensor holdout and group validation. The regression split import remains a compatibility entry point. |
| Converter | UI/session behavior stays in the App; serialization and numerical input contracts are callable without Streamlit. Recognized instrument provenance headers are not measurements; malformed measurement rows fail. |
| Large modules | Validation tables, assessment plots and PDF assembly group related operations. Cohesive responsibilities take priority over a line-count target. |
| Errors | Input and export outer boundaries report failures with context. Expected missing plot data is distinct from computation failure; neither produces fabricated measurements or predictions. |
| Rerun cost | Cached numerical stages are separate from rendering. Further cache/fragment changes require measured rerun costs. |
| Product decisions | Deeper navigation integration, new screening qualification rules and deployment remain separate user decisions; maintenance does not adopt them. |

Active checks are `pytest tests`, `ruff check apps src tests`, and formatting checks for affected code. Legacy tests depend on retired imports and are not part of this active suite. Numerical regression tests use independent expected quantities; artifact presence and a passing suite do not independently validate model generalization or archived scientific claims.

## Evaluation and scientific interpretation

These are retrospective sensor-ID holdout results. Recorded IDs do not prove physical-device or independent biological-preparation identity. Fixed peak choices and estimator policies retain prior dataset-informed decisions. QA describes labeled response consistency; it does not establish prospective sensor qualification or the physical cause of a signal decline. Grouped evaluation and strict fit boundaries improve the estimand's integrity without reproducing historical emailed accuracies or establishing external generalization.

## Review coverage and verification boundary

The active inventory contains 72 library and 25 App Python modules, including package export files. Responsibilities, imports, callable boundaries, input/failure semantics and affected callers are reviewed across this surface. Detailed calculations and revised contracts are supported by independent regression tests; this is software maintenance acceptance, not scientific acceptance of every archived result.

| Area | Review depth |
| --- | --- |
| Loading → metadata/normalization/alignment → scalar/fixed/dynamic features → filters | Detailed source and caller review; invalid-input and independent numerical expectations; complete bundled-data smoke |
| QA/identity preparation → classifiers/global/two-stage/MTL → Validation | Detailed cohort, split, scaler, selection, class-support and failure contracts; outer-test perturbation and grouped inner-split regressions |
| Inventory, assessment/peak orchestration, config/utilities, App composition/cache/reset | Responsibilities and caller review; numerical and lifecycle tests where consequential; complete default App rerun |
| Converter metadata/template UI and serialization | Full extraction/caller review, finite/grid/preset regressions, real instrument TXT/workbook round trip and prep-App export lifecycle |
| Plot factories, report services, PDF layouts, tab presentation/downloads | Detailed data-availability, labeling, figure ownership, complete-table, aspect-ratio and stale-download review; generated PDF text and rendered-page checks |

Verification uses `pytest tests`, Ruff, formatter checks and repository pre-commit hooks. Legacy tests depend on retired imports and are outside this active suite. The active suite currently has 178 passing tests without warnings; completion checks are recorded in Git and the owning project State.

The bundled pipeline loads 125 workbooks /580 spectra, trims to 249 points at 450–1800 cm⁻¹ with exposure normalization, and produces finite integral areas matching an independent segment-by-segment trapezoidal sum. Identity-eligible cohorts contain 580 classification /454 positive-CFU regression rows; these counts cover all default categories/dates. Recent ST/SE (dates >=2025-09-01) remains 259 spectra. These unscreened eligibility counts are distinct from the diagnostic screened cohort.

A real default AppTest renders 16 main/subtabs with no uncaught exception, error or warning. Four instrument TXT files merge on a common grid, retain 231 points at 560–1800 cm⁻¹, and round-trip actual concentrations and explicit sample labels through the workbook loader. Six report types plus a 45-row/13-column stress table are generated; all 32 rendered pages are inspected. The model-layout checks use a synthetic sensor-balanced cohort with default split seeds and four MTL epochs for QA, and do not represent new research performance experiments. PDF failure/availability and download invalidation tests cover error paths separately from those successful outputs.
