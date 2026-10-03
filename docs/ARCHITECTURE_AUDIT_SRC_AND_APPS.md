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

`apps/txt_to_excel.py` contains both the prep UI and parsing/workbook-export logic. Its current single-file form reflects deferred extraction, not a permanent architecture exception. Structural extraction is pending LJ's review; narrow converter correctness fixes do not require treating that layout as the intended long-term design.

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
- Dynamic peak discovery and shared PCA are fitted on the derived input before the classification/regression split. The current classification cleaning also uses labeled responses before splitting. Those paths support exploration but need a training-only evaluation design before scores represent a prospective held-out workflow.
- PCA replaces missing/non-finite spectral values with zero; several model paths fill missing feature values with zero. Missing measurement, unavailable feature, and an undetected peak are different cases. Their scientific policy needs review rather than a generic fallback replacement.

## Maintenance questions

| Area | Evidence and next decision |
| --- | --- |
| Model evaluation | Define the intended independent unit, training-only preprocessing, screening, and model selection together. Avoid repairing PCA in isolation while leaving other leakage paths unchanged. |
| Converter structure | Parsing and serialization are independently useful, but extraction is deferred. Keep its current UI/workbook behavior stable while choosing whether it remains a tool or becomes a separate package. |
| Large modules | `validation_metrics`, `assessment_plots`, and `pdf_builder` group several related operations. Split only when responsibilities or repeated changes justify it; line count alone is insufficient. |
| Error boundaries | Workbook loading records per-file failures; upload and PDF UI boundaries display failures. Broad catches at these outer boundaries are observable rather than silent. PDF failures retain a traceback through logging. |
| PDF figure omissions | Sensor-assessment report preparation currently skips some plot `ValueError`s. Missing sections need an explicit report policy before those failures can be interpreted as complete assessment. |
| Rerun cost | Cached numerical stages are separated from rendering. Profile actual reruns before adding more caching, fragments, or session abstractions. |
| Session reset | `clear_app_data` clears caches and session state. Its lifecycle is deliberate; namespacing is useful only if unrelated state needs preservation. |

Active checks are `pytest tests`, `ruff check apps src tests`, and formatting checks for affected code. Legacy tests depend on retired imports and are not part of this active suite. Numerical regression tests use independent expected quantities; artifact presence and a passing suite do not independently validate model generalization or archived scientific claims.

## Evaluation paths requiring a coordinated design

| Current path | Consequence to resolve before prospective claims |
| --- | --- |
| PCA/dynamic anchors fitted on the derived cohort before splitting | Held-out observations influence representation. Dynamic anchors also use serotype and nominal-dose labels. |
| QA and pooled residual screening before splitting | The evaluated cohort depends on labeled concentration-response evidence. Define what screening evidence would be available for a future sample/sensor. |
| Classification stratifies rows; regression/Validation hold out sensor IDs | These answer different generalization questions. Repeated spectra can share a sensor across classification train/test sets; recorded IDs do not establish physical-device or biological independence. |
| Outer-training scaling precedes tuning CV | Inner validation folds influence the scaler used by the search. Use training-fold transformations when redesigning tuning. |
| MTL early stopping splits training rows and scales using all outer-training rows | Within-training validation can share sensors and influence scaling. The tiny-training fallback can overlap training/validation rows; outer test sensors remain separate under the application split. |
| Classification/two-stage select by held-out F1; global regression selects by held-out RMSE | The same holdout informs model choice and displayed performance. Keep final assessment separate from selection in a future design. |
| Missing features filled with zero | Unavailable measurements and measured zeros share a representation. Define missingness and feature availability together with evaluation. |

## Review coverage and verification boundary

The active inventory contains 67 library and 25 App Python modules, including package export files. Inventory, imports, callable boundaries, error/imputation sites, and test routing are scanned across that surface. This is not a line-by-line acceptance of all 92 modules.

| Area | Review depth |
| --- | --- |
| Workbook loading → metadata/normalization/grid alignment → scalar/peak features → filtering | Detailed source and caller review, independent input/quantity regressions, bundled-data smoke |
| QA/classification/concentration preparation → model/split code → Validation tables/UI | Detailed identity, array/split, error handling, and reproducibility review; focused regressions |
| Inventory, assessment/peak orchestration, configuration/utilities, plotting/report boundaries, App composition/cache/reset | Interface and selected calculation/caller review; existing focused coverage retained |
| Converter metadata/template UI, the full plotting/PDF layout surface, other tab presentation branches | Not fully reviewed line by line; converter extraction and broader navigation/layout work remain outside this pass |

The current checkpoint passes 155 active tests with five existing warnings (degenerate PCA fixtures and small search spaces), Ruff, and formatting for affected code. The bundled upload pipeline loads all 125 default workbooks/580 spectra, trims to 249 points at 450–1800 cm⁻¹ with normalization enabled, and retains finite integral areas matching an independent segment-by-segment trapezoidal sum. Its 412 classification and 322 regression rows retain aligned source indices; this smoke does not train new models. Counts describe all default categories/dates, not the restricted recent ST/SE cohort. Full live-App reruns, every PDF layout, archived accuracy reproduction, and prospective scientific validity are not established by these checks.
