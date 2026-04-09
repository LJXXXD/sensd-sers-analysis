# Architectural Audit: `src/sensd_sers_analysis` and `apps/`

**Scope:** Software architecture, separation of concerns, Streamlit execution patterns, and structural code quality.
**Out of scope:** Changing or critiquing mathematical formulas, ML hyperparameters, or scientific algorithm choices; those are assumed correct for this review.
**Codebase name:** The installable package is `sensd_sers_analysis` (repository: `sensd-sers-analysis`).

---

## 1. Exhaustive Architectural Map

### 1.1 Package layout (high level)

| Area | Role |
|------|------|
| `src/sensd_sers_analysis/` | Library: I/O, preprocessing, features, assessment statistics, classification, visualization, PDF reports |
| `apps/` | Streamlit “SERS Data Explorer”: upload, sidebar controls, filters, tabbed UI |

**Import boundary:** `apps/` imports `sensd_sers_analysis.*` and local `components/`, `tabs/`, `theme.py`. The package does not import `apps/`.

---

### 1.2 `src/sensd_sers_analysis/__init__.py`

- **Purpose:** Public package surface for non-UI consumers (notebooks, scripts, tests).
- **Exports:** Re-exports selected symbols from `assessment`, `data`, `report`, `visualization`; defines `__version__ = "0.1.0"`.
- **Gap vs. app usage:** The Streamlit app relies heavily on `processing` (feature extraction, filters, peaks, PCA) which is **not** re-exported from the root `__init__.py`. That is appropriate for a slim public API but means the “canonical” entrypoint for full pipelines is submodule-based.

---

### 1.3 `src/sensd_sers_analysis/data/`

#### `data/__init__.py`

- Re-exports: `RS_COL_PREFIX`, `count_unique_spectra`, `get_metadata_columns`, `get_raman_shift`, `get_signals_matrix`, `load_sers_data`, `load_sers_data_as_wide_and_tidy`, `wide_to_tidy`.

#### `data/io.py` — core I/O and wide/tidy model

- **Constants / schema:** `REQUIRED_METADATA_KEYS`, `CONCENTRATION_KEY_PATTERN`, `DEFAULT_PATTERN`, `META_COLS`, `RAMAN_SHIFT_DECIMALS`, `RS_COL_PREFIX`.
- **Private helpers:**
  - `_parse_embedded_format(file_path)` → `(metadata_dict, raman_shift ndarray, signals ndarray, concentrations list)`.
  - `_metadata_get(metadata, *keys)` — tolerant key lookup.
  - `_load_signal_file(file_path)` → one wide `DataFrame` (metadata columns + `rs_*` intensity columns).
  - `_collect_files(paths, pattern)` — resolves files/folders to Excel paths.
  - `_get_raman_columns(df)` — sorted `rs_*` column names.
- **Public functions:**
  - `load_sers_data(paths, *, serotypes, pattern)` → concatenated wide `DataFrame`.
  - `get_signals_matrix(df)` → `(n_samples, n_raman_points)` float array.
  - `get_raman_shift(df)` → 1D float array parsed from `rs_*` names.
  - `get_metadata_columns(df)` → metadata-only `DataFrame`.
  - `wide_to_tidy(df)` → long format with `raman_shift`, `intensity`, metadata id columns.
  - `load_sers_data_as_wide_and_tidy(paths, ...)` → `(wide, tidy)`.
  - `count_unique_spectra(df)` — counts unique `(filename, signal_index)`.
- **Data structures:** Wide rows = one row per sample (per signal); spectral grid encoded as column names. Tidy rows = one row per `(sample, raman_shift)` pair.

---

### 1.4 `src/sensd_sers_analysis/processing/`

#### `processing/__init__.py`

- Aggregates public API: `trim_raman_shift`; `extract_basic_features`, feature column helpers; `preprocess_metadata` and related; `filter_*` family; `PeakWindowInfo`, `extract_dynamic_peak_features`, `get_peak_height_columns`; `add_pca_features` (re-exported).

#### `processing/metadata.py`

- **Internal constants:** `_CONC_GROUP_CENTERS_LOG`, `_CONC_GROUP_LABELS`, `_CONC_CATEGORIES` (ordered categorical domain).
- **Functions:**
  - `_extract_scalar_concentration(series, df)` — **row-wise Python loop** over `concentration` when list-valued.
  - `add_log_concentration(df)` → adds `log_concentration`.
  - `add_concentration_group(df)` → adds ordered `concentration_group`.
  - `preprocess_metadata(df)` — applies log + group + date normalization (`YYYY-MM-DD` string).

#### `processing/alignment.py`

- `trim_raman_shift(wide_df, min_shift, max_shift)` — drops `rs_*` columns outside window; preserves metadata.

#### `processing/filters.py`

- **Constants:** `DEFAULT_PLOT_HUE_ORDER`, `DEFAULT_FILTER_ORDER`, `NON_FILTER_COLS`.
- **Functions:**
  - `_filter_mask(df, col, selected, exclude)` — include/exclude semantics.
  - `get_filter_options(df, columns, filter_state)` — **cascading** option lists.
  - `filter_sers_data(df, filter_state)` — compound boolean mask (returns `df.loc[mask]` view).
  - `filter_by_selections(df, selections, invalid_values=...)` — exact-match subset for assessment tabs.
  - `get_filterable_columns(df)` — metadata columns excluding spectra.
  - `get_plot_hue_columns(df)` — ordered columns for plot grouping.
  - `get_feature_metadata_columns(df, feature_cols=None)` — columns that are not feature columns (lazy import of `BASIC_FEATURE_COLUMNS` to avoid cycles).
  - `pick_preferred_column(available, preferred=...)`.

#### `processing/features.py`

- **Constants:** `BASIC_FEATURE_COLUMNS`, `PREFERRED_FEATURE_ORDER`, `DEFAULT_GLOBAL_QA_FEATURES`, `PHASE2_FEATURE_BASE`.
- **Functions:**
  - `get_available_feature_columns(df, peak_infos_by_serotype)` — merges basic + dynamic peak columns in preferred order.
  - `order_features_by_preference(features, preference=None)`.
  - `extract_basic_features(df_wide)` — `nanmax`/`nanmean`, trapezoidal integral via `scipy.integrate.trapezoid`, joins PCA columns from `add_pca_features`.

#### `processing/pca_features.py`

- `add_pca_features(df_wide, *, n_components=2)` — `StandardScaler` + `sklearn.decomposition.PCA`; returns `PC1`, `PC2`, variance ratio columns aligned to `df_wide.index`.

#### `processing/peak_features.py`

- **Constants:** `ZERO_CFU_LABEL`, `_HIGH_CONC_PREFERENCE`, `_DEFAULT_SEROTYPE_PREFERENCE`, `OUTER_BASELINE_FRAC`, `ANCHOR_PROMINENCE_FRAC`, `ANCHOR_DISTANCE_INDICES`.
- **Dataclass:** `PeakWindowInfo` — `peak_name`, `center`, `window_min`, `window_max`, `success_rate`.
- **Private pipeline:** zero-CFU masks, high-concentration subset selection, Savitzky–Golay smoothing, `scipy.signal.find_peaks`, window boundary logic, per-row peak height extraction with baseline subtraction.
- **Public:**
  - `extract_dynamic_peak_features(df_wide, n_peaks=6, *, n_peaks_by_serotype, ...)` → `(peak_df, peak_by_sero, mean_by_sero, default_serotype, raman_x)`.
  - `get_peak_height_columns(peak_infos)`.

---

### 1.5 `src/sensd_sers_analysis/assessment/`

#### `assessment/__init__.py`

- Re-exports batch variance, model consistency (regression QA), consistency tables, degradation, outliers.

#### `assessment/consistency.py`

- **Constant:** `ASSESSMENT_GROUP_COLS`.
- **Dataclass:** `ConsistencyResult` (not always instantiated directly; metrics returned as `Series`/rows).
- **Functions:** `coefficient_of_variation`, `compute_consistency_metrics` (groupby + `filter_outliers`), `get_consistency_summary_table`.

#### `assessment/outliers.py`

- `detect_outliers_iqr`, `detect_outliers_zscore`, `filter_outliers` → `(inliers, outliers)`.

#### `assessment/degradation.py`

- **Dataclass:** `DegradationResult`.
- **Functions:** `prepare_degradation_data` (groupby mean + `test_ordinal`), `compute_degradation` (grouped `scipy.stats.linregress`), `add_sequence_column`.

#### `assessment/batch_variance.py`

- `compute_batch_variance` — per-sensor aggregates + batch z-scores.
- `identify_deviating_sensors` — threshold on `|z_from_batch|`.

#### `assessment/model_consistency.py`

- **Dataclasses:** `ConcentrationRegressionResult`, `CleanedRegressionResult`, `MacroRegressionResult`.
- **Functions:**
  - `fit_concentration_regression`, `fit_concentration_regression_cleaned` (two-pass + residual IQR).
  - `get_zero_cfu_baseline`.
  - `get_global_model_consistency` — raw regression grid over sensor × serotype × feature.
  - `get_global_model_consistency_qa` — cleaned regression + dual threshold + `excluded_map: dict[tuple[str,str], set[str]]`.
  - `compute_macro_batch_regression` — pools inliers across pass sensors.

---

### 1.6 `src/sensd_sers_analysis/classification/`

#### `classification/__init__.py`

- Exports: `prepare_phase2_data`, `train_classifiers`, `ClassificationResult`, plotting helpers.

#### `classification/data_prep.py`

- `prepare_phase2_data(df, *, excluded_map, feature_cols, inlier_feature, ...)` — integrates Phase 1 QA map, rinsate vs pathogen labeling, inlier retention via `fit_concentration_regression_cleaned`.

#### `classification/models.py`

- **Dataclass:** `ClassificationResult` — holds fitted model, metrics, confusion matrix, optional importances/scaler.
- `train_classifiers(df, feature_cols, target_col, test_size=0.2, random_state=42)` — RF + SVM, stratified split.

#### `classification/plots.py`

- `plot_pca_classification`, `plot_confusion_matrix`, `plot_feature_importance` — seaborn/matplotlib figures.

---

### 1.7 `src/sensd_sers_analysis/visualization/`

#### `visualization/__init__.py`

- Re-exports assessment plots, `plot_spectra` + `VARIANCE_OPTIONS`, `plot_feature_distribution`.

#### `visualization/plots.py`

- **Constants:** column name constants, `DEFAULT_NUMERIC_CMAP`, `VARIANCE_OPTIONS` (label, `show_variance`, `errorbar` spec).
- `plot_spectra(df, *, hue, style, show_variance, errorbar, ...)` — seaborn lineplot with units/errorbar semantics; validates tidy schema.

#### `visualization/stats.py`

- `plot_feature_distribution` — box/violin + optional stripplot; concentration ordering via `order_concentration_labels`.

#### `visualization/assessment_plots.py`

- `plot_degradation_trend`, `plot_batch_boxplot`, `plot_concentration_regression`, `plot_multi_sensor_regression`, `plot_macro_batch_regression` (imports assessment fitting utilities for overlays).

---

### 1.8 `src/sensd_sers_analysis/report/`

#### `report/__init__.py`

- Exports `build_sensor_assessment_pdf`, `build_phase1_qa_pdf`, `build_phase2_classification_pdf`.

#### `report/pdf_builder.py`

- ReportLab pipeline: `_df_to_table_data`, `_compute_table_col_widths`, `_figure_to_image_bytes`, plus builders that assemble multi-section PDFs from tables and matplotlib figures.

---

### 1.9 `src/sensd_sers_analysis/utils/`

#### `utils/__init__.py`

- Exports `format_column_label`, `natural_sort`/`natural_sort_key`, `order_concentration_labels`, `parse_raman_shift_bound`.

#### `utils/labels.py`

- `format_column_label` — title-case with acronym handling (`id`, `cfu`, etc.).

#### `utils/natural_sort.py`

- Natural sort keying and `order_concentration_labels`.

#### `utils/parsing.py`

- `parse_raman_shift_bound` — tolerant float parsing for UI strings.

---

### 1.10 `apps/` — Streamlit application

#### `apps/app.py` (composition root / “main”)

- **Responsibilities:** `st.set_page_config`; sidebar upload + “Reload Data”; orchestrates:
  1. `load_from_uploaded` → `(wide_df, tidy_df)`
  2. `preprocess_metadata` on tidy and wide
  3. `render_raman_and_peaks_sidebar` → trim bounds + peak counts
  4. `trim_raman_shift` → `wide_to_tidy` → `preprocess_metadata` again on tidy
  5. `extract_basic_features`, `extract_dynamic_peak_features`, join peak columns into `features_df`
  6. Writes peak context into `st.session_state` (`peak_infos_by_serotype`, `mean_spec_by_serotype`, `peak_default_serotype`, `raman_x`)
  7. Renders filter UI (`get_filterable_columns`, `get_filter_options`, `_render_filter`, `filter_sers_data`)
  8. Dispatches six tabs: `spectra_viewer`, `peak_diagnostics`, `feature_analysis`, `sensor_assessment`, `model_consistency`, `serotype_classification`

#### `apps/theme.py`

- UI constants: Streamlit widths, default figsizes, slider min/max/defaults, peak count limits, matplotlib alpha/fontsize, HTML `<hr>` snippets.

#### `apps/components/data_loading.py`

- `UPLOADER_RESET_KEY`, `clear_app_data()` (clears `st.cache_data` and **all** `session_state`), `load_from_uploaded` (`@st.cache_data`) — writes uploads to temp files, calls `load_sers_data_as_wide_and_tidy`.

#### `apps/components/filter_ui.py`

- `MAIN_FILTER_COUNT`, `FLAT_OPTIONS_THRESHOLD`, `_clear_single_filter`, `_render_filter` (pills vs multiselect), `render_main_filter_header`, `section_divider`.

#### `apps/components/raman_sidebar.py`

- `render_raman_and_peaks_sidebar` — text inputs for min/max shift; derives serotypes from `wide_df`; per-serotype `st.number_input` for peak counts.

#### `apps/components/shared_ui.py`

- `render_pdf_download_section` — injects CSS, Generate/Download pattern using `session_state`.
- `render_metrics_row`, `render_dataframe_stretch`, `render_figure_stretch` (`st.pyplot` + `plt.close`).

#### `apps/components/__init__.py`

- Re-exports `load_from_uploaded`, `MAIN_FILTER_COUNT`, `section_divider` for convenience.

#### `apps/tabs/spectra_viewer.py`

- Widgets for hue/style/variance/height; calls `plot_spectra` + `render_figure_stretch`.

#### `apps/tabs/peak_diagnostics.py`

- Reads peak artifacts from `session_state`; builds **matplotlib figures inline** (mean spectrum with anchors/windows; single-spectrum verification); uses `get_raman_shift`, `get_signals_matrix`; builds diagnostic `pd.DataFrame` from `PeakWindowInfo` fields.

#### `apps/tabs/feature_analysis.py`

- Validates non-NaN basic features; widgets for feature/x/hue/plot type/height; `plot_feature_distribution`.

#### `apps/tabs/sensor_assessment.py`

- Widgets for serotype/concentration/feature/outlier method; `filter_by_selections`; runs consistency, degradation, batch variance pipelines; `_build_assessment_pdf` duplicates computations for PDF; `render_pdf_download_section`.

#### `apps/tabs/model_consistency.py`

- Per-sensor regression UI; global QA table via `get_global_model_consistency_qa`; nested loops for overlay + macro plots; accumulates `overlay_items` / `macro_items` for PDF; Phase 1 PDF generation.

#### `apps/tabs/serotype_classification.py`

- Phase 2 gating on columns + peak columns; calls `get_global_model_consistency_qa` with **`integral_area` only** to build `excluded_map`; `prepare_phase2_data`; `train_classifiers` + plots + Phase 2 PDF.

#### `apps/tabs/__init__.py`

- Re-exports tab modules.

---

### 1.11 Primary data flows (end-to-end)

1. **Upload → wide/tidy:** Excel bytes → temp paths → `load_sers_data` → `wide_to_tidy` (in loader and again after trim).
2. **Metadata enrichment:** `preprocess_metadata` adds `log_concentration`, `concentration_group`, normalized `date`.
3. **Spectral windowing:** `trim_raman_shift` on wide → rebuild tidy (because `rs_*` columns change).
4. **Feature matrix (sample-level):** `extract_basic_features` (macro features + PCA) + `extract_dynamic_peak_features` (peak heights) → left-joined on index in `app.py`.
5. **Filtering:** `filter_sers_data` applied independently to **tidy** spectra (`filtered`) and **feature** table (`filtered_features`) with the same `filter_state` (must stay index-aligned).
6. **Assessment / QA:** Subsets of `filtered_features` by explicit selections; regression and outlier logic in `assessment`.
7. **Phase 2:** `excluded_map` from global QA → `prepare_phase2_data` → `train_classifiers`.

---

## 2. Separation of Concerns (SoC) Violation Audit

### 2.1 “Fat” composition root: `apps/app.py`

**Issue:** The main module is simultaneously:

- Application entrypoint and layout,
- **Data pipeline orchestrator** (trim → tidy → preprocess → feature extraction → peak extraction → session state mutation),
- **Filter engine driver** (iterative `get_filter_options` calls interleaved with widget rendering).

**Heavy operations executed at import/runtime top-level on every Streamlit rerun** (after upload): see pipeline roughly in ```93:205:apps/app.py``` (preprocess, trim, `wide_to_tidy`, `extract_basic_features`, `extract_dynamic_peak_features`, filtering).

**Refactor direction (architecture only):** Introduce a dedicated **application service** module (still in `apps/` or in `src` under e.g. `sensd_sers_analysis/app_pipeline.py`) exposing pure functions or a small class, e.g. `build_explorer_artifacts(uploaded_files, raman_bounds, peak_config) -> ExplorerArtifacts`, and keep `app.py` to wiring only. *Do not change the internal math of the called library functions.*

---

### 2.2 Domain plotting and array logic embedded in `apps/tabs/peak_diagnostics.py`

**Issue:** The tab contains substantial **figure construction** and **numerical indexing** that is not reusable from CLI/tests:

- Mean spectrum plot with `axvline` / `axvspan` per `PeakWindowInfo` (```62:93:apps/tabs/peak_diagnostics.py```).
- Diagnostic table construction (```95:104:apps/tabs/peak_diagnostics.py```).
- Signal-level plot: mask construction, local max for star markers, baseline/window logic intertwined with UI-selected rows (```221:274:apps/tabs/peak_diagnostics.py```).

**Decouple by moving (same matplotlib outputs, same inputs):**

- Functions such as `plot_peak_anchor_diagnostic(raman_x, mean_spec, peak_infos, **style_from_theme)` and `plot_single_spectrum_peak_verification(x, y, row_peak_infos, filtered_features_row, ...)` into `src/sensd_sers_analysis/visualization/` (or a new `visualization/peak_diagnostics_plots.py`).
- Keep the tab limited to: load `session_state` keys, `st.selectbox` values, call visualization + `render_figure_stretch`.

---

### 2.3 PDF generation duplicates on-screen computations: `apps/tabs/sensor_assessment.py`

**Issue:** On-screen path computes consistency table, degradation table/figure, batch table/figure. `_build_assessment_pdf` **recomputes** the same structures for PDF bytes (```253:335:apps/tabs/sensor_assessment.py```).

**SoC impact:** Business rules drift risk (two call sites for one report); doubled CPU on PDF generation; harder testing.

**Decouple:** A single “assessment bundle” builder in `src` (e.g. `sensd_sers_analysis/assessment/report_payload.py`) returning a dataclass with tables + figures + metadata, consumed by both `st` rendering and `build_sensor_assessment_pdf`. *Same underlying functions (`get_consistency_summary_table`, etc.) — just one orchestration path.*

---

### 2.4 Phase 1 tab mixes orchestration, plotting, and PDF assembly: `apps/tabs/model_consistency.py`

**Issues:**

- Computes `global_qa_tbl, excluded_map = get_global_model_consistency_qa(...)` unconditionally when prerequisites pass (```173:176:apps/tabs/model_consistency.py```).
- Nested loops over serotypes × features build plots **and** collect `overlay_items` / `macro_items` for PDF (```221:282:apps/tabs/model_consistency.py```), mixing UX concerns with report payload construction.

**Decouple:** Extract functions:

- `compute_phase1_view_model(filtered_features, selections) -> Phase1ViewModel` (tables, per-combo plot specs),
- `build_phase1_pdf_bytes(view_model)` wrapping `build_phase1_qa_pdf`.

The tab should render from the view model; PDF uses the same artifact container.

---

### 2.5 Phase 2 tab encodes workflow policy: `apps/tabs/serotype_classification.py`

**Issues:**

- Calls `get_global_model_consistency_qa(filtered_features, feature_cols=["integral_area"])` **only** (```74:77:apps/tabs/serotype_classification.py```). That policy (“Phase 2 cleanliness follows integral_area QA map”) is embedded in the UI layer.
- User-facing message says to “Run Model-Based Sensor Consistency first” but Phase 2 does **not** consume cached results from that tab — it recomputes QA (```60:65:apps/tabs/serotype_classification.py``` vs. ```173:176:apps/tabs/model_consistency.py```).

**Decouple:** Move “Phase 2 prerequisites + excluded map source” into `src` (e.g. `classification.phase2_bridge.get_phase2_exclusion_map(df, strategy=...)`) so UI only passes `filtered_features` and explicit user choices. *Same default feature list — just not hardcoded only in Streamlit.*

---

### 2.6 Broad exception handling in UI: `apps/components/shared_ui.py`

**Issue:** `render_pdf_download_section` catches **`Exception`** around `generate_callback()` (```74:83:apps/components/shared_ui.py```), which masks programming errors and breaks fail-fast debugging.

**Architectural fix:** Catch **specific** report-building errors (e.g. `ValueError`, `IOError`) at the boundary, or let the callback raise after logging; keep UI thin.

---

### 2.7 Cross-layer import smell: `classification/data_prep.py`

**Issue:** Imports `_extract_scalar_concentration` from `processing.metadata` — a **private** symbol (leading underscore) used across subsystems (```16:16:src/sensd_sers_analysis/classification/data_prep.py```).

**SoC fix:** Promote to a documented public helper (e.g. `extract_scalar_concentration_per_row` on `metadata` or `data`) without changing its behavior.

---

## 3. State Management & Streamlit Anti-Patterns

### 3.1 Session state as a hidden global data bus

**Peak pipeline outputs** are stored in `st.session_state` in `app.py` (e.g. `peak_infos_by_serotype`, `mean_spec_by_serotype`, `raman_x`, `peak_default_serotype`) — see ```124:144:apps/app.py```. Tabs read these globals (`peak_diagnostics`, `feature_analysis`, `sensor_assessment`, `model_consistency`, `serotype_classification`).

**Risks:**

- Implicit contract between modules (key names duplicated as string literals).
- Harder to unit-test tabs without Streamlit runtime.
- No typed “application state” object.

**Improvement:** A small `ExplorerState` dataclass (or `TypedDict`) owned by a single module, with explicit serialization keys if needed; tabs receive state via parameters rather than reading arbitrary keys when feasible.

---

### 3.2 Full session wipe on reload

`clear_app_data` deletes **every** `session_state` key and clears `st.cache_data` (```22:34:apps/components/data_loading.py```).

**Anti-pattern:** Destructive reset breaks extensibility (future keys from other features vanish) and surprises users who expected only data reload.

**Improvement:** Namespaced keys under `st.session_state["sers_explorer"]` and delete only that subtree; or track keys explicitly.

---

### 3.3 Filter widget keys coupled to display labels

`_render_filter` uses `key=label` where `label` is `format_column_label(col)` (```71:87:apps/components/filter_ui.py```). `Reset all filters` iterates the same labels (```105:108:apps/components/filter_ui.py```).

**Risk:** If `format_column_label` changes, widget identity changes → selections reset unexpectedly.

**Improvement:** Use stable keys `f"filter__{col}"` for widgets; keep `format_column_label` for display only.

---

### 3.4 Redundant recomputation on every rerun

**Always recomputed after upload (no `st.cache_data` on pipeline stages):**

- `preprocess_metadata` twice on overlapping frames (tidy + wide, then tidy again post-trim).
- `wide_to_tidy` after every trim.
- `extract_basic_features` (includes PCA fit per full wide frame).
- `extract_dynamic_peak_features` (per-serotype peak learning + per-row extraction).
- `get_filter_options` called **per filter column** and again inside “More Filters” expander — each rebuilds masks from scratch (```161:192:apps/app.py``` + expander loop).

**Expensive tab bodies:**

- `model_consistency.render`: global QA + potentially many overlay/macro plots each rerun.
- `serotype_classification.render`: `get_global_model_consistency_qa` + `train_classifiers` + multiple plots each rerun.

**Improvement (non-algorithmic):**

- `st.cache_data` on pure steps keyed by `(files_fingerprint, trim_bounds, peak_params_json, filter_state_hash)` — careful invalidation when filters change.
- Split UI with `st.fragment` (Streamlit ≥1.33) so slider changes in one tab do not rerun unrelated heavy sections.
- Defer `train_classifiers` until a “Run training” button (explicit user intent); cache results on `(df_hash, feature_list, random_state)`.

---

### 3.5 Inefficient / fragile data alignment assumptions

- `peak_diagnostics` uses `wide_df.loc[filtered_features.index]` (```113:115:apps/tabs/peak_diagnostics.py```). This assumes **index alignment** between wide and feature tables as loaded/joined in `app.py`. If future refactors reindex or reset, this will break silently.

**Improvement:** Explicit merge keys (`filename`, `signal_index`) in a library helper used by both spectra and diagnostics.

---

### 3.6 PDF artifact accumulation without lazy evaluation

`model_consistency.render` builds `overlay_items` and `macro_items` lists while drawing plots (```221:266:apps/tabs/model_consistency.py```). Even users who never click “Generate Report” pay the cost of **constructing PDF-oriented structures** (figure handles stored).

**Improvement:** Build PDF collections only inside the generate callback (or cache figures separately).

---

## 4. Code Smells & Architectural Debt

### 4.1 Duplication (DRY)

| Area | Description |
|------|-------------|
| Sensor assessment | On-screen vs `_build_assessment_pdf` duplicate orchestration (Section 2.3). |
| Phase 1 / Phase 2 | Repeated column-presence guards (`sensor_id`, `serotype`, `log_concentration`, etc.) across tabs with similar `st.warning` text. |
| Metadata parsing | `_extract_scalar_concentration` private import from classification (Section 2.7). |

---

### 4.2 Magic numbers and policy constants in UI

Examples:

- `identify_deviating_sensors(..., z_threshold=2.0)` in `sensor_assessment` (```209:211:apps/tabs/sensor_assessment.py```) — should be config-driven (YAML/TOML) for **operational** tuning without touching algorithms.
- `FLAT_OPTIONS_THRESHOLD = 50` in `filter_ui.py` (```24:24:apps/components/filter_ui.py```).
- Theme holds reasonable UI constants, but assessment thresholds live inline.

---

### 4.3 Performance hotspots (structural, not formula changes)

- `metadata._extract_scalar_concentration` uses Python `for i in range(len(series))` (```30:38:src/sensd_sers_analysis/processing/metadata.py```) — architectural debt for large tables; vectorization is an implementation efficiency change **without** altering outputs if done carefully.
- `extract_dynamic_peak_features` uses per-row Python loop for sample extraction (```386:410:src/sensd_sers_analysis/processing/peak_features.py```) — same note.
- `get_global_model_consistency_qa` triple nested loops over sensors × serotypes × features (```447:473:src/sensd_sers_analysis/assessment/model_consistency.py```) — consider batching/grouped operations **as engineering**, preserving outputs.

---

### 4.4 Defensive programming gaps

- `shared_ui.render_pdf_download_section`: bare `Exception` (Section 2.6).
- `peak_diagnostics`: assumes `filtered_features` index ⊆ `wide_df.index` (Section 3.5).
- `serotype_classification`: messaging implies ordering dependency on another tab, but code path recomputes QA independently (Section 2.5) — **logical UX inconsistency**, not math.

---

### 4.5 Visualization ↔ assessment coupling

`visualization/assessment_plots.py` imports fitting functions from `assessment.model_consistency` (```15:20:src/sensd_sers_analysis/visualization/assessment_plots.py```). This creates a dependency from visualization layer into core analytics.

**Debt:** Harder to swap plotting backend or run headless analytics without matplotlib/scipy plotting side effects.

**Mitigation pattern:** Pass precomputed regression results into plotting functions (some functions already accept `regression_result=` — extend that pattern).

---

### 4.6 Public API drift

Root `sensd_sers_analysis/__init__.py` does not expose `processing`, while the Streamlit app depends on it heavily. This is acceptable but should be documented for external consumers (architecture doc / README index), not necessarily changed.

---

## 5. Actionable Refactoring Blueprint

**Constraint:** All steps preserve calling the **same** library functions with the **same** arguments unless the change is purely structural (caching keys, file layout, delegation). No intentional changes to numeric outputs, thresholds’ default values, or model definitions.

### Phase A — Stabilize boundaries (low risk)

1. **Introduce `sensd_sers_analysis/pipeline/explorer.py` (or similar)** with a pure dataclass `ExplorerArtifacts`:
   - Fields: `wide_df`, `tidy_df`, `features_df`, `filtered_tidy`, `filtered_features`, `peak_by_sero`, `mean_by_sero`, `default_sero`, `raman_x`, `filter_state`, diagnostics metadata.
   - Methods/functions: `compute_artifacts(upload: UploadSignature, ui_params)`.
2. **Move stringly-typed `session_state` keys** into one module `apps/state_keys.py` as constants.
3. **Fix filter widget keys** to stable column-based identifiers (Section 3.3).

### Phase B — Remove duplication & tighten SoC

4. **Sensor assessment single orchestrator:** One function returns `(tables, figures, pdf_bytes_or_builder)`; tab calls it once, renders tables/figures, passes a closure referencing cached results to `render_pdf_download_section`.
5. **Peak diagnostics plots** moved to `src` visualization module; tab only selects data (Section 2.2).
6. **Promote `_extract_scalar_concentration`** to a public metadata helper (Section 2.7).

### Phase C — Streamlit execution efficiency (behavior-preserving)

7. **Add `st.cache_data` layers** for:
   - Loaded wide/tidy (already present via `load_from_uploaded`),
   - `(wide_df_fingerprint, trim_min, trim_max)` → trimmed wide + tidy,
   - `(trimmed_fingerprint, peak_params)` → `(features_df, peak_dicts, raman_x)`.
8. **Defer ML training** behind a button + `session_state` cache of `ClassificationResult` objects keyed by data hash + feature list.
9. **Use `st.fragment`** on pure control widgets that should not trigger full heavy rerun (where Streamlit version permits).

### Phase D — Interface between `apps/` and `src/`

10. **Define a narrow “application port”** (protocol / abstract base class) e.g. `SersExplorerBackend` with methods:
    - `load_uploads`, `apply_spectral_window`, `build_features`, `apply_filters`, `assessment_bundle`, `phase1_qa_bundle`, `phase2_bundle`.
    - Default implementation delegates to existing `sensd_sers_analysis` functions.
    - Streamlit layer depends only on this interface → eases future FastAPI/CLI parity.

11. **Testing strategy:** Unit-test `ExplorerArtifacts` builders without Streamlit; snapshot **shapes** and **deterministic** metrics on fixture Excel files.

---

## Appendix A — File inventory checklist

**`src/sensd_sers_analysis` (30 files):** all Python modules under `data/`, `processing/`, `assessment/`, `classification/`, `visualization/`, `report/`, `utils/`, and package `__init__.py` — each summarized in Section 1.

**`apps` (14 files):** `app.py`, `theme.py`, `components/{__init__,data_loading,filter_ui,raman_sidebar,shared_ui}.py`, `tabs/{__init__,spectra_viewer,peak_diagnostics,feature_analysis,sensor_assessment,model_consistency,serotype_classification}.py`.

---

## Appendix B — Key session state keys (current)

| Key | Written in | Read in |
|-----|------------|---------|
| `_uploader_reset` | `data_loading.clear_app_data` | `app.py` uploader key |
| `peak_infos_by_serotype` | `app.py` | `peak_diagnostics`, `feature_analysis`, `sensor_assessment`, `model_consistency`, `serotype_classification` |
| `mean_spec_by_serotype` | `app.py` | `peak_diagnostics` |
| `peak_default_serotype` | `app.py` | `peak_diagnostics` |
| `raman_x` | `app.py` | `peak_diagnostics` |
| `assessment_pdf`, `phase1_qa_pdf`, `phase2_pdf` | `shared_ui.render_pdf_download_section` | same (download buttons) |
| Filter widgets | `filter_ui._render_filter` / reset | implicit via widget state |

---

*End of audit document.*
