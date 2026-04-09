# Deep Architectural Audit: `src/` and `apps/`

## Scope
- Focus area: software architecture, separation of concerns, clean code, dependency direction, state management, and execution efficiency.
- Primary directories audited: `src/sensd_sers_analysis/` and `apps/`.
- Explicitly out of scope: changing mathematical formulas, scientific logic, statistical criteria, feature semantics, or ML behavior. The goal of this report is to preserve algorithmic output while improving structure.

## High-Level Assessment
The codebase already has a meaningful split between reusable library code in `src/sensd_sers_analysis/` and the Streamlit front end in `apps/`, which is a good starting point. Most core numerical work is not buried directly inside widgets. However, the current architecture still has a major structural weakness: orchestration is concentrated in `apps/app.py` and several `apps/tabs/*.py` files, so the Streamlit layer is acting as both view layer and application service layer. That creates avoidable rerun costs, weakens reuse, and makes it hard to introduce stable interfaces between UI and backend.

The backend itself is broadly organized into sensible domains:
- `data/` for file parsing and wide/tidy conversion
- `processing/` for metadata derivation, trimming, filters, PCA, and peak features
- `assessment/` for QA and regression-based sensor analysis
- `classification/` for Phase 2 data prep, models, and plots
- `visualization/` for plotting
- `report/` for PDF generation
- `utils/` for formatting, parsing, and natural sorting

The main debt is not that the project lacks modules; it is that there is no explicit application/service layer between `apps/` and `src/`. The Streamlit entrypoint therefore wires low-level functions together directly, repeats work on rerun, and passes derived artifacts through raw `st.session_state` keys instead of through typed contracts.

## 1. Exhaustive Architectural Map

### 1.1 Layered topology

| Layer | Current location | Responsibility | Observed issue |
| --- | --- | --- | --- |
| Presentation | `apps/` | Streamlit widgets, tab layout, user messaging, download buttons | Also performs application orchestration and some data/plot preparation |
| Application orchestration | Mostly implicit inside `apps/app.py` and tab modules | Load data, derive features, route filtered views into analyses | Missing dedicated service layer |
| Domain processing | `src/sensd_sers_analysis/processing/` | Metadata prep, trimming, filters, PCA, peak extraction | Good separation overall, but some duplicated helper patterns |
| Assessment / QA | `src/sensd_sers_analysis/assessment/` | Consistency, batch variance, degradation, regression QA | Some duplicated regression/outlier responsibilities |
| Classification | `src/sensd_sers_analysis/classification/` | Phase 2 clean-data prep, model training, result plotting | Depends on private helper from another layer |
| Visualization | `src/sensd_sers_analysis/visualization/` | Matplotlib/seaborn figures | Some visualization functions still recompute domain data |
| Reporting | `src/sensd_sers_analysis/report/` | PDF assembly | Report builders are reusable, but artifact preparation still lives in tabs |
| Infrastructure / utilities | `src/sensd_sers_analysis/data/`, `utils/` | File parsing, labels, parsing, sort helpers | Good base; could be made more uniform |

### 1.2 End-to-end runtime data flow

The current runtime flow is:

1. `apps/app.py` defines the page, upload widget, and reload button.
2. Uploaded files are converted to `(filename, bytes)` tuples and sent to `apps/components/data_loading.py:load_from_uploaded()`.
3. `load_from_uploaded()` writes each uploaded file to a temporary directory and delegates parsing to `src/sensd_sers_analysis/data/io.py:load_sers_data_as_wide_and_tidy()`.
4. `data/io.py` parses Excel files into a wide dataframe and a tidy dataframe:
   - wide form: metadata columns plus `rs_*` spectral columns
   - tidy form: one row per `(sample, raman_shift)` pair
5. `apps/app.py` runs `preprocess_metadata()` on both `wide_df` and `tidy_df`.
6. `apps/components/raman_sidebar.py` collects Raman window and per-serotype peak-count settings.
7. `apps/app.py` applies `trim_raman_shift()` to `wide_df`, rebuilds tidy data with `wide_to_tidy()`, and re-runs `preprocess_metadata()` on the new tidy dataframe.
8. `apps/app.py` computes:
   - `features_df = extract_basic_features(wide_df)`
   - `peak_df, peak_by_sero, mean_by_sero, default_sero, raman_x = extract_dynamic_peak_features(...)`
9. `apps/app.py` joins peak-height columns into `features_df` and pushes peak artifacts into `st.session_state`.
10. `apps/app.py` derives filterable columns from the tidy dataframe, renders filter widgets, computes cascading options, and applies filters to both tidy spectra and feature-level data.
11. Tab renderers consume:
   - `filtered` for spectra display
   - `filtered_features` for feature analysis, assessment, model QA, and classification
   - peak artifacts from `st.session_state` for diagnostics and feature availability
12. Tab modules call backend plotting/report/model functions on demand.

### 1.3 Primary in-memory data structures

These data structures are the real application contract today:

| Name | Shape / type | Produced in | Consumed by |
| --- | --- | --- | --- |
| `wide_df` | `pd.DataFrame` with metadata + `rs_*` columns | `data/io.py`, then trimmed in `app.py` | `extract_basic_features`, `extract_dynamic_peak_features`, `peak_diagnostics` |
| `tidy_df` | `pd.DataFrame` with metadata + `raman_shift` + `intensity` | `data/io.py`, regenerated in `app.py` after trim | filter UI, spectra viewer |
| `features_df` | metadata + scalar features + PCA + peak heights | `app.py` | most tabs |
| `peak_df` | metadata + `Peak_i_Height` columns | `extract_dynamic_peak_features()` | joined into `features_df` |
| `peak_infos_by_serotype` | `dict[str, list[PeakWindowInfo]]` | `app.py` via peak extraction | `peak_diagnostics`, feature availability in tabs |
| `mean_spec_by_serotype` | `dict[str, np.ndarray]` | `app.py` via peak extraction | `peak_diagnostics` |
| `raman_x` | `np.ndarray` | `app.py` via peak extraction | `peak_diagnostics` |
| `excluded_map` | `dict[tuple[str, str], set[str]]` | model consistency QA functions | model consistency plots, Phase 2 data prep |
| `phase2_clean` | cleaned feature dataframe with `target` | `classification/data_prep.py` | `train_classifiers`, PCA/classification plots |
| `ClassificationResult` | dataclass | `classification/models.py` | confusion matrix / feature-importance rendering |

### 1.4 `src/` exhaustive file-by-file map

#### `src/sensd_sers_analysis/__init__.py`
- Purpose: curated root package API for non-UI consumers.
- Exports:
  - assessment helpers such as `compute_batch_variance`, `compute_degradation`
  - data functions such as `load_sers_data`, `wide_to_tidy`
  - visualization functions such as `plot_spectra`, `plot_batch_boxplot`
  - only one report builder: `build_sensor_assessment_pdf`
- Core observation: the root API does not expose `processing` or `classification`, so the app imports those subpackages directly. This is reasonable for a slim public API, but it means there is no single "official" pipeline facade for the UI to use.

#### `src/sensd_sers_analysis/data/__init__.py`
- Purpose: re-export of the data-loading/public I/O surface.
- Core symbols: `RS_COL_PREFIX`, `count_unique_spectra`, `get_metadata_columns`, `get_raman_shift`, `get_signals_matrix`, `load_sers_data`, `load_sers_data_as_wide_and_tidy`, `wide_to_tidy`.

#### `src/sensd_sers_analysis/data/io.py`
- Purpose: the primary I/O boundary for embedded-metadata Excel files.
- Core constants:
  - `REQUIRED_METADATA_KEYS`
  - `CONCENTRATION_KEY_PATTERN`
  - `DEFAULT_PATTERN`
  - `META_COLS`
  - `RAMAN_SHIFT_DECIMALS`
  - `RS_COL_PREFIX`
- Core private helpers:
  - `_parse_embedded_format()` parses one Excel file and returns normalized metadata, Raman shift values, signal matrix, and concentrations.
  - `_metadata_get()` resolves flexible metadata key names.
  - `_load_signal_file()` constructs the wide dataframe for one file.
  - `_collect_files()` resolves mixed file/folder inputs.
  - `_get_raman_columns()` discovers and sorts `rs_*` columns.
- Core public functions:
  - `load_sers_data()`
  - `get_signals_matrix()`
  - `get_raman_shift()`
  - `get_metadata_columns()`
  - `wide_to_tidy()`
  - `load_sers_data_as_wide_and_tidy()`
  - `count_unique_spectra()`
- Core data structures:
  - wide dataframe is the canonical analysis format
  - tidy dataframe is the canonical plotting/filtering format
- Dependencies:
  - `numpy`, `pandas`, `pathlib`, `logging`
- Architectural role:
  - This is one of the best-separated modules in the project: it owns file-format parsing and dataframe construction cleanly.

#### `src/sensd_sers_analysis/utils/__init__.py`
- Purpose: convenience export module for utility helpers.
- Exports: `format_column_label`, `natural_sort`, `natural_sort_key`, `order_concentration_labels`, `parse_raman_shift_bound`.

#### `src/sensd_sers_analysis/utils/labels.py`
- Purpose: format technical column names for user-facing display.
- Core function: `format_column_label(col: str) -> str`.
- Core data structure: `_LABEL_ACRONYMS`.
- Architectural note: this helper is used by the app to derive widget labels, which means UI state keys become coupled to display text.

#### `src/sensd_sers_analysis/utils/natural_sort.py`
- Purpose: human-friendly ordering for strings with embedded numbers.
- Core functions:
  - `natural_sort_key()`
  - `natural_sort()`
  - `order_concentration_labels()`
- Used by:
  - `processing/filters.py`
  - `processing/metadata.py`
  - `visualization/plots.py`
  - `visualization/stats.py`
- Architectural note: this is a useful shared utility and already serves as a small SSOT for concentration label ordering.

#### `src/sensd_sers_analysis/utils/parsing.py`
- Purpose: parse UI string input for Raman bounds.
- Core function: `parse_raman_shift_bound(s: str) -> float | None`.
- Architectural note: despite living in `src`, this is primarily a Streamlit-input adapter rather than a domain function.

#### `src/sensd_sers_analysis/processing/__init__.py`
- Purpose: broad export surface for processing functions and constants.
- Exposes:
  - alignment
  - basic feature extraction
  - filters and filter metadata
  - metadata preprocessing
  - PCA features
  - dynamic peak extraction and `PeakWindowInfo`
- Architectural note: this is the package the app uses most heavily, but there is no higher-level "pipeline" object built on top of it.

#### `src/sensd_sers_analysis/processing/alignment.py`
- Purpose: trim Raman shift windows while preserving metadata.
- Core function: `trim_raman_shift()`.
- Depends on: `data.RS_COL_PREFIX`.
- Good boundary: pure dataframe transformation, no UI dependency.

#### `src/sensd_sers_analysis/processing/features.py`
- Purpose: compute global scalar features and coordinate feature availability/order.
- Core constants:
  - `BASIC_FEATURE_COLUMNS`
  - `PREFERRED_FEATURE_ORDER`
  - `DEFAULT_GLOBAL_QA_FEATURES`
  - `PHASE2_FEATURE_BASE`
- Core functions:
  - `get_available_feature_columns()`
  - `order_features_by_preference()`
  - `extract_basic_features()`
- Dependencies:
  - `data.get_signals_matrix()`
  - `data.get_raman_shift()`
  - `processing.pca_features.add_pca_features()`
  - `processing.peak_features.get_peak_height_columns()`
- Architectural note:
  - This module is a coordination point for feature semantics.
  - It also acts as a feature-schema SSOT for multiple downstream modules.
  - It still hardcodes `"rs_"` in metadata-column exclusion instead of reusing `RS_COL_PREFIX`.

#### `src/sensd_sers_analysis/processing/filters.py`
- Purpose: discover filterable metadata columns and apply selection/exclusion masks.
- Core constants:
  - `DEFAULT_PLOT_HUE_ORDER`
  - `DEFAULT_FILTER_ORDER`
  - `NON_FILTER_COLS`
  - `_DEFAULT_INVALID_SELECTIONS`
- Core functions:
  - `_filter_mask()`
  - `get_filter_options()`
  - `filter_sers_data()`
  - `filter_by_selections()`
  - `get_filterable_columns()`
  - `get_plot_hue_columns()`
  - `get_feature_metadata_columns()`
  - `pick_preferred_column()`
- Dependencies:
  - `utils.natural_sort.order_concentration_labels()`
  - `data.RS_COL_PREFIX`
- Architectural note:
  - Strong candidate for moving behind a higher-level filter service.
  - `get_feature_metadata_columns()` uses a function-local import of `BASIC_FEATURE_COLUMNS`, which is a code smell indicating a circular dependency boundary.

#### `src/sensd_sers_analysis/processing/metadata.py`
- Purpose: add derived metadata required for downstream QA and grouping.
- Core constants:
  - `_CONC_GROUP_CENTERS_LOG`
  - `_CONC_GROUP_LABELS`
  - `_CONC_CATEGORIES`
- Core functions:
  - `_extract_scalar_concentration()`
  - `add_log_concentration()`
  - `add_concentration_group()`
  - `preprocess_metadata()`
- Dependencies:
  - `numpy`, `pandas`
  - `utils.natural_sort.natural_sort()`
- Architectural note:
  - `_extract_scalar_concentration()` is private but reused by `classification/data_prep.py`, which is a boundary violation.
  - This module is effectively part of the domain-data contract, not just metadata cleanup.

#### `src/sensd_sers_analysis/processing/pca_features.py`
- Purpose: append PCA coordinates and explained-variance ratios from spectral columns.
- Core function: `add_pca_features()`.
- Dependencies:
  - `sklearn.decomposition.PCA`
  - `sklearn.preprocessing.StandardScaler`
  - `data.get_signals_matrix()`
- Architectural note:
  - Cleanly isolated, though called eagerly on every rerun from `app.py`.

#### `src/sensd_sers_analysis/processing/peak_features.py`
- Purpose: dynamic, serotype-aware peak discovery and per-sample peak-height extraction.
- Core constants:
  - `ZERO_CFU_LABEL`
  - `_HIGH_CONC_PREFERENCE`
  - `_DEFAULT_SEROTYPE_PREFERENCE`
  - `OUTER_BASELINE_FRAC`
  - `ANCHOR_PROMINENCE_FRAC`
  - `ANCHOR_DISTANCE_INDICES`
- Core dataclass:
  - `PeakWindowInfo`
- Core helper functions:
  - `_is_zero_cfu()`
  - `_exclude_zero_cfu()`
  - `_find_high_conc_subset()`
  - `_smooth_spectrum()`
  - `_find_peaks_on_spectrum()`
  - `_find_outer_left()`
  - `_find_outer_right()`
  - `_compute_peak_windows_for_serotype()`
  - `_pick_default_serotype()`
- Core public functions:
  - `extract_dynamic_peak_features()`
  - `get_peak_height_columns()`
- Architectural note:
  - This is one of the most complex modules and should stay in `src`.
  - It also hardcodes `"rs_"` when excluding spectral columns from metadata.
  - Runtime cost is driven by per-row loops and per-window scans.

#### `src/sensd_sers_analysis/assessment/__init__.py`
- Purpose: export surface for assessment and QA functions.
- Exports:
  - batch variance
  - consistency
  - degradation
  - outlier helpers
  - regression QA dataclasses and functions
- Architectural note: this is the domain API that most assessment tabs should depend on, but the tabs still orchestrate too much themselves.

#### `src/sensd_sers_analysis/assessment/batch_variance.py`
- Purpose: compute per-sensor summary statistics and identify deviating sensors.
- Core functions:
  - `compute_batch_variance()`
  - `identify_deviating_sensors()`
- Dependencies:
  - `numpy`, `pandas`
- Architectural note: well-contained domain logic with clear dataframe input/output.

#### `src/sensd_sers_analysis/assessment/consistency.py`
- Purpose: compute raw vs outlier-filtered CV and related feature consistency metrics.
- Core constant:
  - `ASSESSMENT_GROUP_COLS`
- Core dataclass:
  - `ConsistencyResult`
- Core functions:
  - `coefficient_of_variation()`
  - `compute_consistency_metrics()`
  - `get_consistency_summary_table()`
- Dependencies:
  - `assessment.outliers.filter_outliers()`
  - `processing.BASIC_FEATURE_COLUMNS`
- Architectural note:
  - `ConsistencyResult` is defined but not used as a returned contract anywhere in the codebase.

#### `src/sensd_sers_analysis/assessment/degradation.py`
- Purpose: prepare temporal grouping and compute linear degradation trends.
- Core dataclass:
  - `DegradationResult`
- Core functions:
  - `prepare_degradation_data()`
  - `compute_degradation()`
  - `add_sequence_column()`
- Dependencies:
  - `numpy`, `pandas`, `scipy.stats`
- Architectural note:
  - `DegradationResult` is similarly defined but not used as the outward API.
  - `prepare_degradation_data()` is a good orchestration helper and should be consumed through an application service, not from the tab directly.

#### `src/sensd_sers_analysis/assessment/model_consistency.py`
- Purpose: perform regression-based model QA and pooled macro-regression.
- Core dataclasses:
  - `ConcentrationRegressionResult`
  - `CleanedRegressionResult`
  - `MacroRegressionResult`
- Core private function:
  - `_detect_residual_outliers_iqr()`
- Core public functions:
  - `fit_concentration_regression()`
  - `fit_concentration_regression_cleaned()`
  - `get_zero_cfu_baseline()`
  - `get_global_model_consistency()`
  - `get_global_model_consistency_qa()`
  - `compute_macro_batch_regression()`
- Dependencies:
  - `numpy`, `pandas`, `scipy.stats`
  - `processing.BASIC_FEATURE_COLUMNS`
- Architectural note:
  - This is the core analytical engine for the model-consistency tab.
  - It duplicates IQR logic conceptually with `assessment/outliers.py`, though on residuals.

#### `src/sensd_sers_analysis/assessment/outliers.py`
- Purpose: generic outlier detection and dataframe splitting.
- Core functions:
  - `detect_outliers_iqr()`
  - `detect_outliers_zscore()`
  - `filter_outliers()`
- Dependencies:
  - `numpy`, `pandas`
- Architectural note:
  - Good as a shared primitive.
  - The `detect_outliers_zscore()` docstring says robust statistics but the implementation uses mean and standard deviation.

#### `src/sensd_sers_analysis/classification/__init__.py`
- Purpose: public Phase 2 export surface.
- Exports:
  - `prepare_phase2_data`
  - `ClassificationResult`
  - `train_classifiers`
  - classification plot builders

#### `src/sensd_sers_analysis/classification/data_prep.py`
- Purpose: produce a strict clean dataset for Phase 2 classification.
- Core function: `prepare_phase2_data()`.
- Dependencies:
  - `assessment.fit_concentration_regression_cleaned()`
  - `assessment.get_global_model_consistency_qa()`
  - private helper `processing.metadata._extract_scalar_concentration()`
- Architectural note:
  - Important cross-layer coupling issue: this module imports a private function from another package.
  - The function also embeds a fixed rule that the inlier gate feature is `integral_area` unless overridden.

#### `src/sensd_sers_analysis/classification/models.py`
- Purpose: train baseline Phase 2 classifiers and package the results.
- Core dataclass:
  - `ClassificationResult`
- Core function:
  - `train_classifiers()`
- Dependencies:
  - `sklearn.ensemble.RandomForestClassifier`
  - `sklearn.svm.SVC`
  - `sklearn.model_selection.train_test_split`
  - `sklearn.preprocessing.StandardScaler`
  - sklearn metrics
- Architectural note:
  - Good functional boundary.
  - Model/training parameters are hardcoded here instead of centralized in config.

#### `src/sensd_sers_analysis/classification/plots.py`
- Purpose: plot PCA class scatter, confusion matrix, and RF feature importances.
- Core functions:
  - `plot_pca_classification()`
  - `plot_confusion_matrix()`
  - `plot_feature_importance()`
- Dependencies:
  - `matplotlib`
  - `seaborn`
  - `ClassificationResult`
- Architectural note:
  - Clean plotting-only module.

#### `src/sensd_sers_analysis/visualization/__init__.py`
- Purpose: export spectral, assessment, and statistics plots.
- Exports:
  - `VARIANCE_OPTIONS`
  - `plot_spectra`
  - `plot_feature_distribution`
  - `plot_batch_boxplot`
  - `plot_concentration_regression`
  - `plot_degradation_trend`
  - `plot_macro_batch_regression`
  - `plot_multi_sensor_regression`

#### `src/sensd_sers_analysis/visualization/plots.py`
- Purpose: render tidy spectral data with optional grouping and variance overlays.
- Core constants:
  - `RAMAN_SHIFT_COL`
  - `INTENSITY_COL`
  - `FILENAME_COL`
  - `SIGNAL_INDEX_COL`
  - `DEFAULT_NUMERIC_CMAP`
  - `VARIANCE_OPTIONS`
- Core functions:
  - `plot_spectra()`
  - `_prepare_continuous_hue()`
  - `_format_errorbar_text()`
  - `_hue_to_label()`
  - `_validate_data()`
  - `_apply_aesthetics()`
  - `_filter_legend_to_style_only()`
  - `_add_colorbar()`
- Dependencies:
  - `matplotlib`, `seaborn`, `pandas`
  - `utils.order_concentration_labels()`
- Architectural note:
  - A clean presentation module.

#### `src/sensd_sers_analysis/visualization/stats.py`
- Purpose: render box/violin/strip plots for scalar feature distributions.
- Core function:
  - `plot_feature_distribution()`
- Dependencies:
  - `matplotlib`, `seaborn`, `pandas`
  - `utils.order_concentration_labels()`
- Architectural note:
  - Another clean plotting-only module.

#### `src/sensd_sers_analysis/visualization/assessment_plots.py`
- Purpose: render degradation, batch variance, regression QA, multi-sensor overlays, and macro regression plots.
- Core functions:
  - `plot_degradation_trend()`
  - `plot_batch_boxplot()`
  - `plot_concentration_regression()`
  - `plot_multi_sensor_regression()`
  - `plot_macro_batch_regression()`
- Dependencies:
  - `assessment.model_consistency` functions and dataclasses
  - `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy.stats`
- Architectural note:
  - `plot_macro_batch_regression()` redoes pooled-inlier collection before calling `compute_macro_batch_regression()`. That is duplicated orchestration inside a visualization module.

#### `src/sensd_sers_analysis/report/__init__.py`
- Purpose: export PDF builders.
- Exports:
  - `build_sensor_assessment_pdf`
  - `build_phase1_qa_pdf`
  - `build_phase2_classification_pdf`

#### `src/sensd_sers_analysis/report/pdf_builder.py`
- Purpose: assemble ReportLab PDFs from precomputed tables and figures.
- Core helpers:
  - `_df_to_table_data()`
  - `_compute_table_col_widths()`
  - `_figure_to_image_bytes()`
- Core public builders:
  - `build_sensor_assessment_pdf()`
  - `build_phase1_qa_pdf()`
  - `build_phase2_classification_pdf()`
- Dependencies:
  - `reportlab`
  - `pandas`
  - `pathlib`, `datetime`, `io`
- Architectural note:
  - Strong reusable boundary.
  - There is still duplication of styling and table assembly patterns within this file.

### 1.5 `apps/` exhaustive file-by-file map

#### `apps/app.py`
- Purpose: single Streamlit entrypoint for loading data, global preprocessing, filtering, and tab dispatch.
- Core responsibilities currently embedded here:
  - upload widget setup
  - cache-aware load invocation
  - metadata preprocessing
  - Raman trimming
  - wide-to-tidy conversion
  - basic feature extraction
  - dynamic peak extraction
  - session-state propagation of peak artifacts
  - dynamic filter rendering and application
  - tab construction
- Key imported backend functions:
  - `count_unique_spectra`, `wide_to_tidy`
  - `extract_basic_features`, `extract_dynamic_peak_features`
  - `filter_sers_data`, `get_filter_options`, `get_filterable_columns`
  - `get_peak_height_columns`, `preprocess_metadata`, `trim_raman_shift`
- Key data structures:
  - `wide_df`, `tidy_df`, `features_df`, `peak_df`, `filtered`, `filtered_features`, `filter_state`
- Architectural note:
  - This file is the main architectural bottleneck in the app. It is functioning as a controller, service coordinator, and state broker all at once.

#### `apps/theme.py`
- Purpose: UI constants and HTML fragments.
- Core symbols:
  - sizing, slider limits, plotting constants, divider HTML
- Architectural note:
  - Good attempt at centralization for UI-only constants.
  - Backend and cross-cutting thresholds are still scattered elsewhere.

#### `apps/components/__init__.py`
- Purpose: partial re-export of reusable component helpers.
- Exports:
  - `load_from_uploaded`
  - `MAIN_FILTER_COUNT`
  - `section_divider`

#### `apps/components/data_loading.py`
- Purpose: bridge Streamlit uploads to path-based backend loaders.
- Core functions:
  - `clear_app_data()`
  - `load_from_uploaded()`
- Key state:
  - `UPLOADER_RESET_KEY`
- Architectural note:
  - This is a reasonable adapter module.
  - `clear_app_data()` is overly aggressive because it deletes all keys in `st.session_state`.

#### `apps/components/filter_ui.py`
- Purpose: render filter widgets and reset controls.
- Core functions:
  - `_clear_single_filter()`
  - `_render_filter()`
  - `render_main_filter_header()`
  - `section_divider()`
- Key behavior:
  - filter state keys are based on display labels, not canonical column names
  - reset-all calls `st.rerun()`
- Architectural note:
  - This is mostly presentation code, but it contributes to hidden coupling through widget-key design.

#### `apps/components/raman_sidebar.py`
- Purpose: capture Raman min/max bounds and per-serotype peak counts.
- Core function:
  - `render_raman_and_peaks_sidebar()`
- Architectural note:
  - Thin UI adapter, appropriately lightweight.

#### `apps/components/shared_ui.py`
- Purpose: shared rendering helpers for PDFs, metrics, dataframes, and figures.
- Core functions:
  - `render_pdf_download_section()`
  - `render_metrics_row()`
  - `render_dataframe_stretch()`
  - `render_figure_stretch()`
- Architectural note:
  - Helpful shared UI surface.
  - `render_pdf_download_section()` uses broad `except Exception`.

#### `apps/tabs/__init__.py`
- Purpose: aggregate tab modules for import in `app.py`.

#### `apps/tabs/spectra_viewer.py`
- Purpose: view-layer wrapper around `plot_spectra()`.
- Core function:
  - `render(filtered)`
- Architectural note:
  - This is one of the cleaner tabs: it mostly collects view settings and delegates to plotting.

#### `apps/tabs/peak_diagnostics.py`
- Purpose: validate peak windows and per-signal detections.
- Core function:
  - `render(filtered_features, wide_df)`
- Responsibilities embedded here:
  - reads peak artifacts from session state
  - builds diagnostic tables
  - slices wide data by filtered feature indices
  - reconstructs selected spectrum
  - manually searches for local maxima inside windows
  - constructs matplotlib figures inline
- Architectural note:
  - This tab contains the heaviest mix of UI logic and backend-like data manipulation among the tabs.

#### `apps/tabs/feature_analysis.py`
- Purpose: view-layer wrapper around `plot_feature_distribution()`.
- Core function:
  - `render(filtered_features)`
- Architectural note:
  - Mostly thin and acceptable.

#### `apps/tabs/sensor_assessment.py`
- Purpose: render consistency, degradation, batch stability, and PDF report controls.
- Core functions:
  - `render(filtered_features)`
  - `_build_assessment_pdf(...)`
- Responsibilities embedded here:
  - selection-to-subset orchestration
  - calls to consistency/degradation/batch APIs
  - duplicated artifact construction for PDF
- Architectural note:
  - The computational primitives live in `src`, but the orchestration should also move there.

#### `apps/tabs/model_consistency.py`
- Purpose: render per-sensor regression QA, global QA, multi-sensor overlays, macro regression, and Phase 1 PDF.
- Core function:
  - `render(filtered_features)`
- Responsibilities embedded here:
  - selection-to-subset orchestration
  - regression QA invocation
  - QA table generation
  - nested overlay and macro-regression loops
  - PDF artifact collection
- Architectural note:
  - Similar to `sensor_assessment.py`: the tab is acting as both presenter and workflow service.

#### `apps/tabs/serotype_classification.py`
- Purpose: Phase 2 clean-data prep, classifier training, result display, and PDF generation.
- Core function:
  - `render(filtered_features)`
- Responsibilities embedded here:
  - recomputes QA exclusion map
  - prepares clean Phase 2 dataset
  - trains classifiers
  - recomputes plots for PDF callback
- Architectural note:
  - This is the clearest example of ML workflow orchestration sitting directly in the UI layer.

## 2. Separation of Concerns (SoC) Violation Audit

This section focuses only on architectural placement, not on whether the formulas or models are correct.

### 2.1 High-severity SoC violations

| Severity | Location | Evidence | Why this is a SoC violation | Suggested destination in `src/` |
| --- | --- | --- | --- | --- |
| High | `apps/app.py:93-127` | `preprocess_metadata()`, `trim_raman_shift()`, `wide_to_tidy()`, `extract_basic_features()`, `extract_dynamic_peak_features()` are all run in the top-level app script | The entrypoint is performing the full derivation pipeline on every rerun; this is application-service logic, not view logic | New application module such as `src/sensd_sers_analysis/application/dataset_pipeline.py` |
| High | `apps/app.py:151-198` | `get_filterable_columns()`, `get_filter_options()`, and `filter_sers_data()` are orchestrated directly inside the entrypoint | Filter-state derivation and filtered-view construction should be a stable backend contract, not a hand-built script section | `src/sensd_sers_analysis/application/filtering_service.py` |
| High | `apps/tabs/peak_diagnostics.py:113-274` | Direct dataframe alignment, spectrum reconstruction, Raman extraction, argmax search inside windows, manual plotting | This tab contains domain-specific verification logic and figure construction, not just widget collection | Data extraction in `src/sensd_sers_analysis/application/peak_diagnostics_service.py`; plotting in `src/sensd_sers_analysis/visualization/peak_diagnostics.py` |
| High | `apps/tabs/serotype_classification.py:74-116` | `get_global_model_consistency_qa()`, `prepare_phase2_data()`, `train_classifiers()` are executed inside the tab render path | Clean-data preparation and model training are backend workflows and should be wrapped in a service API | `src/sensd_sers_analysis/application/classification_service.py` |

### 2.2 Medium-severity SoC violations

| Severity | Location | Evidence | Why this is a SoC violation | Suggested destination in `src/` |
| --- | --- | --- | --- | --- |
| Medium | `apps/tabs/sensor_assessment.py:104-225` | `filter_by_selections()`, `get_consistency_summary_table()`, `prepare_degradation_data()`, `compute_degradation()`, `compute_batch_variance()`, `identify_deviating_sensors()` are coordinated in the tab | The primitives are in `src`, but the orchestration is duplicated and therefore not reusable outside Streamlit | `src/sensd_sers_analysis/application/assessment_service.py` |
| Medium | `apps/tabs/sensor_assessment.py:253-335` | `_build_assessment_pdf()` recomputes assessment tables and figures already computed in `render()` | Report artifact preparation should be backend orchestration, not a UI callback concern | Same `assessment_service.py`, plus report artifact builder |
| Medium | `apps/tabs/model_consistency.py:104-176` | `fit_concentration_regression_cleaned()`, `get_zero_cfu_baseline()`, `get_global_model_consistency_qa()` are coordinated inside the tab | The tab is acting as a workflow controller instead of consuming a service result object | `src/sensd_sers_analysis/application/model_consistency_service.py` |
| Medium | `apps/tabs/model_consistency.py:221-281` | nested loops across serotypes and features build overlays and macro regressions inline | Cross-product orchestration belongs in a backend service so it can be cached and tested independently | same model-consistency service |

### 2.3 Low-severity or acceptable UI/backend mixing

| Severity | Location | Assessment |
| --- | --- | --- |
| Low | `apps/components/data_loading.py:57-63` | Writing uploaded bytes to temp files is an acceptable adapter concern because the backend expects filesystem paths |
| Low | `apps/tabs/spectra_viewer.py` | Mostly view-only: gathers widget values and calls `plot_spectra()` |
| Low | `apps/tabs/feature_analysis.py` | Mostly view-only: gathers widget values and calls `plot_feature_distribution()` |

### 2.4 Exact frontend code that should be decoupled into `src/`

The most important migrations are:

1. `apps/app.py:93-127`
   - Move all dataframe-derivation steps into one backend call.
   - Proposed API:
     - `build_derived_bundle(loaded_bundle, min_shift, max_shift, n_peaks_by_serotype) -> DerivedDataBundle`

2. `apps/app.py:151-198`
   - Move filter-column discovery, cascading-option computation, and filtered-view construction behind a backend/application layer.
   - Proposed API:
     - `build_filter_catalog(tidy_df) -> FilterCatalog`
     - `apply_filter_state(bundle, filter_state) -> FilteredBundle`

3. `apps/tabs/peak_diagnostics.py:113-274`
   - Move selected-spectrum lookup, local peak-position extraction, and diagnostic plotting preparation into `src`.
   - Proposed API:
     - `build_peak_diagnostic_view(filtered_features, wide_df, peak_artifacts, selection) -> PeakDiagnosticArtifacts`
     - `plot_peak_windows(...)`
     - `plot_signal_level_peak_verification(...)`

4. `apps/tabs/sensor_assessment.py:104-225` and `253-335`
   - Move assessment artifact assembly and PDF-input construction into `src`.
   - Proposed API:
     - `build_sensor_assessment_artifacts(filtered_features, selection) -> SensorAssessmentArtifacts`
     - `build_sensor_assessment_report_inputs(artifacts) -> ...`

5. `apps/tabs/model_consistency.py:104-176` and `221-281`
   - Move model-consistency artifact assembly and overlay batch generation into `src`.
   - Proposed API:
     - `build_model_consistency_artifacts(filtered_features, selection, overlay_request) -> ModelConsistencyArtifacts`

6. `apps/tabs/serotype_classification.py:74-116`
   - Move Phase 2 clean-data derivation and training orchestration into `src`.
   - Proposed API:
     - `build_phase2_dataset(filtered_features, qa_policy) -> Phase2Dataset`
     - `run_phase2_classification(phase2_dataset, feature_policy) -> Phase2Artifacts`

## 3. State Management and Streamlit Anti-Patterns

### 3.1 Current `st.session_state` contract

The app is currently using `st.session_state` as a raw cross-module message bus.

| Key pattern | Set in | Read in | Problem |
| --- | --- | --- | --- |
| `_uploader_reset` | `components/data_loading.py:34` | `app.py:74` | Fine for uploader reset, but global session clearing is too broad |
| `peak_infos_by_serotype` | `app.py:128` / `app.py:138` | `peak_diagnostics.py`, `feature_analysis.py`, `sensor_assessment.py`, `model_consistency.py`, `serotype_classification.py` | Hidden cross-tab dependency with no typed contract |
| `mean_spec_by_serotype` | `app.py:129` / `app.py:139` | `peak_diagnostics.py` | Same issue |
| `peak_default_serotype` | `app.py:130` / `app.py:140` | `peak_diagnostics.py` | Same issue |
| `raman_x` | `app.py:131` / `app.py:141` | `peak_diagnostics.py` | Same issue |
| filter widget keys based on labels | `filter_ui.py` | implicitly by Streamlit widget state and reset handlers | Keys depend on display labels rather than stable column IDs |
| PDF byte buffers such as `assessment_pdf`, `phase1_qa_pdf`, `phase2_pdf` | `shared_ui.py` | `shared_ui.py` | Acceptable, but still raw stringly typed state |

### 3.2 Why the current state model is inefficient

The app stores some derived artifacts in session state, but not the expensive ones that would actually prevent recomputation.

Stored:
- peak metadata and arrays
- uploader reset token
- filter widget values
- PDF byte blobs

Not stored or cached:
- preprocessed `wide_df`
- trimmed `wide_df`
- rebuilt `tidy_df`
- `features_df`
- `peak_df`
- global QA tables
- overlay/macro regression artifacts
- `phase2_clean`
- trained classification results

The result is that session state is being used for communication, not for true computation memoization.

### 3.3 Rerun-driven recomputation map

Because Streamlit reruns the script on any widget change, the following recomputation occurs today:

| User interaction | What reruns | Cost impact |
| --- | --- | --- |
| Change any main filter | Entire `app.py` reruns; metadata prep, Raman trimming, tidy rebuild, basic features, peak extraction, filter-option recomputation, filtering | High |
| Change Raman min/max | Entire derivation pipeline reruns | High |
| Change number of peaks per serotype | Entire derivation pipeline reruns, especially dynamic peak extraction | High |
| Change model-consistency controls | Global top-level pipeline reruns first, then tab-specific QA/plots rerun | High |
| Change Phase 2 controls or generate Phase 2 PDF | Global top-level pipeline reruns; classification prep and training rerun; plots may be regenerated for PDF | High |
| Click Generate report in assessment/model/classification tabs | Rerun plus callback-side artifact recomputation | Medium to high |

### 3.4 Current caching pattern

Only one meaningful cache boundary exists:

- `apps/components/data_loading.py:37` uses `@st.cache_data` on `load_from_uploaded()`.

What is good about this:
- repeated reruns do not reparse uploads if the file tuple is unchanged

What is still missing:
- cache boundary after upload parse but before heavy derived computations
- cache boundary for trimmed/re-featured bundles keyed by Raman window and peak counts
- cache boundary for filtered views keyed by filter state
- cache boundary for QA and Phase 2 artifacts keyed by filtered feature frame + UI selections

### 3.5 Specific Streamlit anti-patterns

1. Raw session keys instead of typed state wrappers
   - The tabs assume specific keys exist and that their payload shapes are correct.
   - There is no dataclass like `PeakArtifactsState` or `AppDataState`.

2. Display-label-based widget keys
   - `apps/components/filter_ui.py` uses formatted labels as state keys.
   - If label formatting changes, state semantics change.
   - If two columns ever produce the same display label, state collisions are possible.

3. `clear_app_data()` wipes all state
   - `apps/components/data_loading.py:31-34` clears every key in `st.session_state`.
   - This is operationally simple but too blunt as the app grows.

4. Cross-tab hidden coupling through session state
   - `feature_analysis`, `sensor_assessment`, `model_consistency`, and `serotype_classification` all indirectly depend on peak metadata being prepared in `app.py`.
   - That dependency is not visible in function signatures.

5. Caching only the cheapest stable boundary
   - Upload parsing is cached.
   - Most expensive derived computations are not.

### 3.6 Inefficient data-loading patterns

The upload path itself is reasonable, but the overall data lifecycle is not optimized:

- `load_from_uploaded()` writes temporary files and delegates to the backend once, which is acceptable.
- `app.py` then preprocesses and re-derives all downstream data every rerun.
- `wide_to_tidy()` is called again after trimming, which is logically correct, but it is done eagerly on every rerun rather than memoized.
- `extract_dynamic_peak_features()` runs on every interaction that causes a rerun, even if the user is only changing a filter in a tab that does not need new peak artifacts.

## 4. Code Smells and Architectural Debt

### 4.1 Duplication and DRY violations

#### Duplicate assessment work for PDF generation
- `apps/tabs/sensor_assessment.py:142-225` computes assessment tables and figures for display.
- `apps/tabs/sensor_assessment.py:253-335` recomputes those same artifacts for PDF generation.
- This is not just repetitive code; it creates two independent orchestration paths that can drift.

#### Duplicate Phase 2 plotting for PDF generation
- `apps/tabs/serotype_classification.py:99-140` renders PCA, confusion matrix, and feature importance.
- `apps/tabs/serotype_classification.py:147-161` regenerates those figures inside the PDF callback.

#### Duplicate macro-regression pooling logic
- `src/sensd_sers_analysis/assessment/model_consistency.py:530-657` builds pooled macro-regression inputs.
- `src/sensd_sers_analysis/visualization/assessment_plots.py:514-562` repeats similar pass-sensor pooling before calling `compute_macro_batch_regression()`.

#### Duplicate/parallel outlier logic
- `src/sensd_sers_analysis/assessment/outliers.py` has generic IQR logic.
- `src/sensd_sers_analysis/assessment/model_consistency.py:100-127` has residual-specific IQR logic.
- The split may be justified semantically, but it still represents duplicated thresholding machinery.

#### Repeated "metadata columns excluding spectral columns" logic
- `src/sensd_sers_analysis/processing/features.py:144-147`
- `src/sensd_sers_analysis/processing/peak_features.py:429-434`
- Both independently derive metadata-column subsets by excluding columns starting with `"rs_"`.

#### Repeated ReportLab styling patterns
- `src/sensd_sers_analysis/report/pdf_builder.py` repeats similar title, heading, table, and color setup across three builders.
- This is manageable now but will become noisy as reports grow.

### 4.2 Poor abstractions and missing interfaces

#### Missing application layer
This is the most important architectural gap.

Current situation:
- `apps/app.py` and tab modules call low-level `src` functions directly.
- There is no stable interface like:
  - `DatasetPipelineService`
  - `FilteringService`
  - `AssessmentService`
  - `ModelConsistencyService`
  - `ClassificationService`

Consequence:
- orchestration logic is duplicated across tabs
- cache boundaries are hard to define
- testing requires driving UI-adjacent code instead of service objects
- the frontend and backend are tightly coupled to dataframe shapes and implicit session keys

#### Private backend helper imported across layers
- `src/sensd_sers_analysis/classification/data_prep.py:16` imports `_extract_scalar_concentration` from `processing.metadata`.
- This breaks encapsulation and signals that the wrong public API is exposed.

#### Non-package-style imports in `apps/`
- Many app files use `from theme import ...`, `from components...`, `from tabs...`.
- This works for the Streamlit execution context but makes `apps/` less portable and less explicit as a package.

#### Visualization modules partly recompute analytics
- `src/sensd_sers_analysis/visualization/assessment_plots.py` is not purely presentational because some functions do analytical setup work before plotting.

### 4.3 Hardcoded magic numbers and configuration scattering

Good news:
- `apps/theme.py` centralizes some view-layer constants cleanly.

Remaining configuration scattering:

In `apps/`:
- `apps/components/filter_ui.py`
  - `MAIN_FILTER_COUNT = 5`
  - `FLAT_OPTIONS_THRESHOLD = 50`
- `apps/components/shared_ui.py`
  - button colors `#28a745`, `#fd7e14`
- `apps/tabs/peak_diagnostics.py`
  - marker size `s=200`
  - manual green/darkgreen colors
  - modulo-9 color cycling
- `apps/tabs/sensor_assessment.py`
  - `z_threshold=2.0`
- `apps/tabs/serotype_classification.py`
  - hardcoded QA feature `["integral_area"]`

In `src/`:
- `classification/models.py`
  - `test_size=0.2`
  - `random_state=42`
  - `n_estimators=100`
  - `min_per_class = 2`
- `assessment/model_consistency.py`
  - `rejection_multiplier=2.0`
  - `r2_min_threshold=0.80`
- `assessment/degradation.py`
  - `rel_slope < 0.5` threshold for "stable"
- `processing/peak_features.py`
  - multiple constants controlling windowing and anchor detection

Architectural impact:
- not all of these numbers are "bad"
- but they are spread across modules with no config SSOT
- captions and text in the UI also encode some of the same thresholds, which creates drift risk

### 4.4 Defensive programming and robustness gaps

#### Broad exception handling
- `apps/components/shared_ui.py:80-83` catches `Exception` broadly in `render_pdf_download_section()`.
- This makes debugging harder and can hide the real failure mode.

#### Upload boundary lacks a UI-specific exception wrapper
- `apps/components/data_loading.py:62-65` warns if the backend returns empty frames.
- But there is no outer UI-specific error boundary around the call itself if parsing raises unexpectedly.

#### Index alignment assumption in peak diagnostics
- `apps/tabs/peak_diagnostics.py:113-115` does:
  - `wide_filtered = wide_df.loc[filtered_features.index]`
- This assumes the indices remain aligned across the wide dataframe and the filtered feature dataframe.
- That is currently true by convention, but it is an implicit contract, not a defended one.

#### Global reset behavior is too coarse
- `apps/components/data_loading.py:31-34` deletes all session state keys.
- That is simple now but risky once more state is added.

#### Dead or misleading symbols
- `src/sensd_sers_analysis/assessment/consistency.py:25-39` defines `ConsistencyResult` but does not use it as the outward API.
- `src/sensd_sers_analysis/assessment/degradation.py:76-86` defines `DegradationResult` but does not use it similarly.
- `src/sensd_sers_analysis/assessment/outliers.py:61-62` says robust statistics are used for Z-score, but the implementation does not do that.

### 4.5 Execution-efficiency debt

#### Top-level rerun pipeline in `app.py`
- The biggest efficiency issue is not inside one function; it is the fact that the entire derivation pipeline is rerun on any control interaction.

#### Python-row loops in backend hot paths
- `src/sensd_sers_analysis/processing/metadata.py:31-38`
  - `_extract_scalar_concentration()` uses a Python loop over rows.
- `src/sensd_sers_analysis/processing/peak_features.py:386-409`
  - sample-by-sample peak-height extraction is a nested Python loop.

These are not necessarily wrong, but because the UI reruns them frequently, their cost becomes much more visible.

#### QA recomputation across tabs
- `get_global_model_consistency_qa()` is executed in:
  - `apps/tabs/model_consistency.py:173-176`
  - `apps/tabs/serotype_classification.py:74-77`
- That means a high-cost QA path is recomputed in multiple tabs instead of being shared.

#### Cascading filter options recalculated repeatedly
- `apps/app.py:161-192`
  - `get_filter_options()` is recomputed inside every filter-render loop iteration.
- This is logically understandable, but on larger datasets it becomes an avoidable cost unless memoized.

#### Figures are sometimes cached indirectly instead of data artifacts
- The current app stores some figure-related results by keeping PDF bytes or lists of figures around, but it does not establish clean data-artifact boundaries that could be reused for multiple render targets.

## 5. Actionable Refactoring Blueprint

This blueprint is intentionally constrained to preserve algorithmic output. The plan is to move and repackage existing logic, not change formulas, model families, or mathematical decision rules.

### 5.1 Target architecture

Introduce an explicit application layer in `src`, for example:

```text
src/sensd_sers_analysis/
  application/
    __init__.py
    contracts.py
    dataset_pipeline.py
    filtering_service.py
    peak_diagnostics_service.py
    assessment_service.py
    model_consistency_service.py
    classification_service.py
```

Suggested contract objects:

- `LoadedDataBundle`
  - `wide_df`
  - `tidy_df`

- `PeakArtifacts`
  - `peak_infos_by_serotype`
  - `mean_spec_by_serotype`
  - `default_serotype`
  - `raman_x`

- `DerivedDataBundle`
  - `wide_df`
  - `tidy_df`
  - `features_df`
  - `peak_df`
  - `peak_artifacts`

- `FilteredBundle`
  - `filtered_tidy_df`
  - `filtered_features_df`
  - `n_unique_spectra`

- `SensorAssessmentArtifacts`
  - consistency table
  - degradation table
  - batch table
  - deviating sensors table
  - figures or figure-ready data

- `ModelConsistencyArtifacts`
  - single-sensor regression result
  - global QA table
  - excluded map
  - overlay requests/results
  - macro regression results

- `Phase2Artifacts`
  - `phase2_clean`
  - `rf_result`
  - `svm_result`
  - best-model pointer

### 5.2 Step-by-step migration plan

#### Step 1: Extract the current top-level dataframe pipeline out of `apps/app.py`

Create `src/sensd_sers_analysis/application/dataset_pipeline.py` with pure functions such as:

- `load_uploaded_bundle(files_data) -> LoadedDataBundle`
- `build_derived_bundle(loaded_bundle, *, min_shift, max_shift, n_peaks_by_serotype) -> DerivedDataBundle`

Implementation rule:
- do not rewrite the underlying math
- internally call the exact same existing functions:
  - `preprocess_metadata`
  - `trim_raman_shift`
  - `wide_to_tidy`
  - `extract_basic_features`
  - `extract_dynamic_peak_features`

Outcome:
- `apps/app.py` becomes a thin coordinator instead of a data-engineering script

#### Step 2: Introduce a stable filter service

Create `src/sensd_sers_analysis/application/filtering_service.py`:

- `build_filter_catalog(tidy_df) -> FilterCatalog`
- `compute_filter_options(tidy_df, filter_columns, filter_state) -> dict[...]`
- `apply_filters(derived_bundle, filter_state) -> FilteredBundle`

Also introduce a typed `FilterState` contract based on canonical column names, not display labels.

Outcome:
- `apps/components/filter_ui.py` remains responsible only for widgets
- `apps/app.py` stops hand-building filtered dataframes

#### Step 3: Replace raw session-state keys with a typed adapter

Create `apps/state.py` or `src/.../application/contracts.py` plus a tiny adapter:

- `read_peak_artifacts_from_state()`
- `write_peak_artifacts_to_state(peak_artifacts)`
- `reset_ui_state()`

Rules:
- keep UI-only ephemeral state in `apps/`
- keep domain artifacts in typed dataclasses
- store those dataclasses or a stable serializable representation in `st.session_state`

Outcome:
- tabs stop depending on string literals like `"peak_infos_by_serotype"`

#### Step 4: Move peak diagnostics logic out of the tab

Split `apps/tabs/peak_diagnostics.py` into:

- UI-only tab:
  - collects serotype/sensor/concentration/signal selections
  - renders returned tables/figures

- Backend services:
  - `build_peak_diagnostic_artifacts(...)`
  - `plot_peak_anchor_summary(...)`
  - `plot_signal_level_peak_verification(...)`

Important constraint:
- preserve the exact same peak-window search and local-maximum behavior
- only move the logic, do not reinterpret it

Outcome:
- peak diagnostics becomes testable without Streamlit

#### Step 5: Create an assessment service

Create `src/sensd_sers_analysis/application/assessment_service.py` with:

- `build_sensor_assessment_artifacts(filtered_features, selection) -> SensorAssessmentArtifacts`
- `build_sensor_assessment_pdf_bytes(artifacts, report_title) -> bytes`

Implementation:
- internally call the same existing assessment and visualization functions
- compute once, then reuse for both on-screen display and PDF

Outcome:
- removes duplication between `render()` and `_build_assessment_pdf()`

#### Step 6: Create a model-consistency service

Create `src/sensd_sers_analysis/application/model_consistency_service.py` with:

- `build_single_sensor_consistency_artifacts(...)`
- `build_global_qa_artifacts(...)`
- `build_overlay_artifacts(...)`
- `build_phase1_pdf_bytes(...)`

Important constraint:
- keep `get_global_model_consistency_qa()` as the analytical core
- move only orchestration and artifact composition

Outcome:
- the tab becomes a thin presenter over service outputs
- overlay and macro-regression results become reusable and cacheable

#### Step 7: Create a classification service

Create `src/sensd_sers_analysis/application/classification_service.py` with:

- `build_phase2_dataset(filtered_features, *, excluded_map_policy, inlier_feature) -> pd.DataFrame`
- `run_phase2_classification(phase2_clean, feature_columns) -> Phase2Artifacts`
- `build_phase2_pdf_bytes(phase2_artifacts) -> bytes`

Rules:
- keep using:
  - `get_global_model_consistency_qa()`
  - `prepare_phase2_data()`
  - `train_classifiers()`
  - existing plot builders
- do not change label semantics or the classifier families

Outcome:
- `apps/tabs/serotype_classification.py` stops orchestrating ML workflow directly

#### Step 8: Add proper cache boundaries around application services

Recommended Streamlit caching boundaries:

- `@st.cache_data`
  - parsed upload bundle keyed by file bytes
  - derived bundle keyed by:
    - uploaded bundle identity
    - Raman min/max
    - per-serotype peak-count settings
  - filtered bundle keyed by:
    - derived bundle identity
    - filter state
  - assessment/model/classification artifacts keyed by:
    - filtered features identity
    - tab-specific selection parameters

Guideline:
- cache data artifacts, not widget code
- cache artifact bundles before plotting when possible

Outcome:
- filter changes no longer force unnecessary feature recomputation if the derived bundle is memoized

#### Step 9: Centralize configuration without changing semantics

Create a config module such as:

- `src/sensd_sers_analysis/config/app_constants.py`
- `src/sensd_sers_analysis/config/model_policies.py`

Move scattered constants there:
- UI-only constants can stay in `apps/theme.py`
- analysis-policy constants should live in `src`
- any UI captions that mention thresholds should read from the same constants used by the backend

Outcome:
- fewer drift points between code, captions, and reports

#### Step 10: Clean cross-layer and schema boundaries

Specific cleanup actions:

1. Replace `classification/data_prep.py` import of `_extract_scalar_concentration`
   - introduce a public helper such as `extract_scalar_concentration()`
   - or move concentration normalization to a dedicated shared schema module

2. Remove hardcoded `"rs_"` checks from modules that should use `RS_COL_PREFIX`
   - especially `processing/features.py`
   - and `processing/peak_features.py`

3. Move shared feature-schema constants into one obvious home if circular imports persist
   - for example `processing/schema.py` or `domain/feature_schema.py`

4. Decide whether `ConsistencyResult` and `DegradationResult` are real public contracts
   - if yes, return them or use them
   - if no, remove them

Outcome:
- clearer contracts and lower maintenance risk

#### Step 11: Preserve output parity during refactor

Because the requirement is to preserve scientific and ML behavior, the refactor should be staged behind characterization tests:

- snapshot dataframe-shape and column-name tests for:
  - `wide_df`
  - `tidy_df`
  - `features_df`
  - `peak_df`
- parity tests for:
  - peak window metadata
  - consistency tables
  - global QA tables
  - Phase 2 target distribution
- UI smoke tests can remain minimal; the important thing is that service outputs remain identical

This is the key guardrail:
- move logic first
- optimize only after the outputs are proven identical

### 5.3 What `apps/` should look like after refactor

After refactor, `apps/` should mostly contain:
- widget declarations
- layout and tab composition
- user-facing captions/errors/success messages
- download buttons
- thin calls into application services

Example target pattern for `app.py`:

```python
loaded_bundle = load_uploaded_bundle(files_data)
derived_bundle = build_derived_bundle(
    loaded_bundle,
    min_shift=min_shift,
    max_shift=max_shift,
    n_peaks_by_serotype=n_peaks_by_serotype,
)
filtered_bundle = apply_filters(derived_bundle, filter_state)
write_peak_artifacts_to_state(derived_bundle.peak_artifacts)
```

That is the architectural shape the current codebase wants, but does not yet have.

### 5.4 Refactor priority order

Recommended order of execution:

1. Extract `dataset_pipeline.py`
2. Add typed state wrapper and filter service
3. Move peak diagnostics logic out of the tab
4. Move sensor-assessment orchestration into a service
5. Move model-consistency orchestration into a service
6. Move Phase 2 orchestration into a service
7. Centralize config and contract cleanup
8. Add characterization tests to lock output parity

This order gives the biggest payoff early:
- `app.py` shrinks first
- rerun cost becomes manageable earlier
- tab code becomes thinner and easier to verify

## Closing Summary

The codebase is not architecturally chaotic, but it is architecturally top-heavy in the Streamlit layer. The backend modules in `src/sensd_sers_analysis/` are already good enough to support a cleaner design; the missing piece is an application-service layer that turns many small low-level functions into stable, cacheable workflows. If that layer is introduced carefully, the project can keep the current scientific behavior exactly as-is while becoming substantially easier to maintain, test, and scale.
