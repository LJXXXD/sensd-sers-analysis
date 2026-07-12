# Architectural audit: `src/sensd_sers_analysis` and `apps/`

**Scope:** Software architecture, separation of concerns, Streamlit execution patterns, and structural code quality.

**Out of scope:** Mathematical formulas, ML hyperparameters, or scientific algorithm choices unless noted as coupling or configuration drift.

**Codebase:** Installable package `sensd_sers_analysis` (repository `sensd-sers-analysis`).

**Companion doc:** [`THEORY_AND_IMPLEMENTATION.md`](THEORY_AND_IMPLEMENTATION.md) describes the scientific pipeline, data model, and module responsibilities for day-to-day use. **This audit is not redundant with that file:** THEORY focuses on *what the pipeline computes*; this document focuses on *layer boundaries, UI coupling, and maintainability debt*.

**History:** An older “deep” audit file (`ARCHITECTURE_AUDIT_SRC_AND_APPS_DETAILED.md`) duplicated most of this material at greater length; it was merged here and retired so a single audit stays current.

---

## 1. Layered topology (current)

| Layer | Location | Responsibility |
| --- | --- | --- |
| Presentation | `apps/` | Streamlit layout, widgets, messaging, downloads |
| Application orchestration | `src/sensd_sers_analysis/application/` | Cached pipelines, filter serialization, PDF/report orchestration, DTOs (`contracts.py`) |
| Domain | `data/`, `processing/`, `assessment/`, `classification/`, `regression/` | Parsing, features, QA, classification, concentration-regression paradigms |
| Visualization | `visualization/` | Matplotlib/seaborn figures (some paths still pull fitting helpers from `assessment/`) |
| Reporting | `report/` | ReportLab PDF assembly from precomputed tables and figures |
| Policy SSOT | `config/` | Thresholds and policy constants (`model_policies.py`, `spectral_policies.py`, `targeted_peaks.py`, …) |
| Utilities | `utils/` | Labels, natural sort, parsing |

**Dependency rule:** Lower layers do not import Streamlit. `apps/` imports `sensd_sers_analysis.application` and domain subpackages as needed.

---

## 2. What improved since earlier audits

These items previously flagged as gaps have largely been addressed:

- **`application/` layer:** `dataset_pipeline.py`, `filtering_service.py`, assessment/sensor/classification/regression services, and typed contracts replace much of the old “fat `app.py` doing everything” pattern.
- **Caching:** `apps/cache.py` memoizes derived bundles, filters, and several assessment/classification/regression artifact builders.
- **Session state:** `apps/state.py` centralizes peak artifact persistence and filter widget keys (`get_filter_widget_key(column)`), avoiding display-label coupling that older audits criticized.
- **Peak diagnostics:** Plotting for peak discovery lives under `visualization/peak_discovery.py` with preparation in `application/peak_discovery_service.py`; tabs delegate rather than owning all matplotlib logic.
- **Legacy Sensor QC PDF:** `sensor_qc_legacy` routes through cached `build_cached_sensor_assessment_artifacts` and `build_sensor_assessment_pdf_bytes`, reducing duplicate orchestration versus the older pattern.
- **Config:** Global QA, serotype-classification policy, batch deviation, and related constants are centralized in `config/model_policies.py` (with ongoing migration noted in THEORY §27).
- **TXT → Excel prep mode:** `apps/txt_to_excel.py` is a separate Streamlit shell (session key `_app_mode`) for merging instrument `.txt` exports into embedded-metadata workbooks. Analysis mode links to it via sidebar **Convert TXT → Excel**; prep does not import domain modules beyond shared visualization.
- **Extended embedded Excel I/O:** `data/io.py` now parses per-signal **File Name** and **Special Treatment** rows, prefers an **Actual Concentration** row when present (falling back to the first concentration-labeled row), and allows blank target CFU cells in prep-generated workbooks.
- **Master-grid Raman alignment:** `processing/alignment.py:snap_spectra_to_master_grid` runs in `build_derived_bundle` before trim/features, aligning heterogeneous sensor grids via overlap-localized linear interpolation (dedupe tolerances in `config/spectral_policies.py`).
- **Empty-state onboarding:** Analysis and prep modes both render title + info + numbered **Next steps** when no files are loaded, reducing first-run confusion without adding domain logic.
- **Prep metadata UX:** Sixteen-field grid layout, JSON preset v1 with selective export, reload snapshot that preserves instrument settings while clearing run-specific IDs, and merged-preview legends that show Target/Actual/source-TXT provenance.

---

## 3. Architectural map (concise)

### 3.1 Package layout

| Area | Role |
| --- | --- |
| `src/sensd_sers_analysis/` | Library: I/O, preprocessing, features, assessment, classification, **regression paradigms**, visualization, PDF reports, application services |
| `apps/` | Streamlit “SERS Data Explorer”: dual-mode shell (analysis + TXT prep), upload, sidebar filters, **ten** analysis tabs |

**Import boundary:** `apps/` imports `sensd_sers_analysis.*`, `cache`, `components`, `tabs`, `theme`, `state`, and (for prep only) `txt_to_excel`. The library does not import `apps/`.

### 3.2 `apps/app.py` (composition root)

**Dual app modes:** When `st.session_state["_app_mode"] == "prep"`, `app.py` calls `render_prep_mode()` from `txt_to_excel.py` and `st.stop()`s before analysis UI. Otherwise it renders `render_prep_entry_in_sidebar` (link into prep mode) and the analysis pipeline below.

Analysis mode orchestrates: upload → `load_from_uploaded` / `LoadedDataBundle` → Raman sidebar bounds → `build_cached_derived_bundle` (includes master-grid snap) → `write_peak_artifacts_to_state` → filter catalog + `apply_cached_filters` → optional `merge_targeted_peaks_into_filtered_bundle` for downstream tabs → tab dispatch.

When no workbook is uploaded, analysis mode renders a branded empty state (`st.title("SERS Data Explorer")`, sidebar-oriented info callout, numbered **Next steps** checklist) instead of stopping silently — same onboarding pattern as prep mode.

Compared to legacy audits, **`app.py` is materially thinner**: derivation and filtering are delegated to `application/` + `cache.py`. Prep orchestration stays in `txt_to_excel.py` (~1.8k lines) by design — it is presentation-layer I/O tooling, not domain logic.

### 3.3 Notable library surfaces

- **`data/io.py`:** Embedded-metadata Excel → wide/tidy; per-signal provenance columns (`source_txt_filename`, `special_treatment`); Actual-vs-Target concentration row selection; `RS_COL_PREFIX`, `META_COLS`, `load_sers_data`, `wide_to_tidy`, etc.
- **`processing/`:** Metadata enrichment, **master-grid snap** + trim (`alignment.py`), filters (including new metadata filter columns), PCA, dynamic peaks (`peak_features.py`), **targeted peaks** (`targeted_peak_features.py`).
- **`assessment/`:** Consistency, outliers, degradation, batch variance, regression-based sensor QA (`sensor_assessment_regression.py`).
- **`classification/`:** Clean-frame prep for serotype ML, RF/SVM training, plots.
- **`regression/`:** Alternative concentration-regression paradigms (global, two-stage, multi-task / `MtlSpectralNet`), splits, metrics, plots — exercised from Streamlit tabs `regression_global`, `regression_two_stage`, `regression_mtl`.
- **`application/`:** Services bridging UI and domain (`*_service.py`), `contracts.py` DTOs, `dataset_pipeline.py`, `merge_targeted_peaks_into_filtered_bundle`, regression PDF builders via `regression_service.py`.
- **`visualization/`:** Spectra, stats, assessment plots, peak discovery plots, targeted peak plots.
- **`report/pdf_builder.py`:** Sensor assessment PDFs, sensor-assessment QA PDF, serotype classification report PDF, **regression** PDF helpers consumed by application services.

---

## 4. Primary data flows (end-to-end)

### 4.1 Instrument TXT → embedded Excel (prep mode)

1. **Upload `.txt` files** in prep sidebar → parse tab-separated instrument exports (`txt_to_excel._parse_txt_content`).
2. **Metadata + per-signal rows:** Sixteen file-level metadata fields (sensor geometry, IDs, date/time, …), optional JSON **metadata preset** import/export, per-file Target/Actual concentration inputs (target may be blank), optional Special Treatment labels.
3. **Merge + export:** Spectra merged on a shared Raman grid within user bounds → `_build_embedded_workbook_rows` → styled `.xlsx` bytes → download; merged preview uses shared `plot_spectra` inside an `@st.fragment` to limit rerun churn.

Prep output workbooks feed the analysis pipeline via normal sidebar upload.

### 4.2 Analysis mode (embedded Excel onward)

1. **Upload → wide/tidy:** Bytes → temp paths → `load_uploaded_bundle` / `load_sers_data_as_wide_and_tidy` (reads per-signal provenance when present).
2. **Derived bundle:** `preprocess_metadata` → **`snap_spectra_to_master_grid`** → trim Raman → basic features + dynamic peaks → `DerivedDataBundle` + `PeakArtifacts`.
3. **Optional targeted peaks:** Session-managed anchor list merged into filtered features for analysis tabs (`merge_targeted_peaks_into_filtered_bundle`).
4. **Filtering:** `FilterCatalog` + serialized `FilterState` → `FilteredBundle` (tidy + features; index alignment preserved).
5. **Assessment / QA / Serotype classification / Regression tabs:** Call cached service builders then render tables and figures.

---

## 5. Remaining separation-of-concerns and debt

### 5.1 Visualization ↔ assessment coupling

`visualization/assessment_plots.py` imports fitting helpers from `assessment.sensor_assessment_regression`. That keeps plotting convenient but binds presentation to regression internals. **Mitigation:** Prefer passing precomputed regression results into plot functions where feasible (pattern already used in places).

### 5.2 Duplicate or parallel numerical machinery

- Residual IQR in `sensor_assessment_regression.py` vs generic IQR in `assessment/outliers.py` — semantically related but separate implementations.
- Macro batch pooling logic has historically appeared in both regression helpers and plotting paths; consolidate orchestration in domain/application layers when touching those modules.

### 5.3 Broad exception handling in PDF UI

`apps/components/shared_ui.py` — `render_pdf_download_section` still catches generic `Exception` around PDF generation. **Prefer:** narrow exceptions plus logging, or re-raise after logging in debug workflows.

### 5.4 Session lifecycle

`clear_app_data` remains a blunt reset (clears caches and session state). Namespaced state would scale better if many unrelated keys accumulate.

### 5.5 Remaining rerun cost

Caching covers major dataframe/service stages; plot construction and some tab bodies still rerun with Streamlit’s execution model. Prep mode already isolates post-convert download/preview in `@st.fragment` (`_render_prep_export_and_preview`); analysis tabs could adopt the same pattern where widget jitter is noticeable. Further gains would come from deferring expensive actions behind buttons or artifact caching keyed more aggressively — without changing scientific outputs.

### 5.6 Package-level hygiene

- Root `__init__.py` exposes a **small** curated API (data, assessment summaries, core plots, one PDF builder); the app correctly imports `processing`, `classification`, `application`, etc. via subpackages.
- **`ConsistencyResult` / `DegradationResult`:** Defined dataclasses are not always the outward API; either adopt them as return types or simplify exports over time.

### 5.7 Documentation drift

Single source for “what exists” should remain aligned with `THEORY_AND_IMPLEMENTATION.md` (modules, tabs, config, prep mode). This audit should be updated when adding new tabs, services, prep/I/O format changes, or cross-cutting concerns.

---

## 6. Refactoring backlog (prioritized, behavior-preserving)

1. **Tighten PDF error boundaries** in `shared_ui` (specific exceptions + structured logging).
2. **Reduce visualization→assessment imports** by threading precomputed results into assessment plots.
3. **Namespace session reset** keys under a single prefix for safer reload behavior.
4. **Consolidate duplicated pooling/IQR orchestration** between plotting and regression QA when next refactoring those files.
5. **Extend characterization tests** beyond application services as critical paths grow (`regression/`, targeted peaks). Recent additions (`test_embedded_io_per_signal_rows.py`, `test_txt_to_excel_template.py`) cover prep/I/O edges but are not yet wired into CI documentation in README.

---

## Appendix A — Streamlit tabs (current)

| Tab | Module | Purpose |
| --- | --- | --- |
| Spectra Viewer | `tabs/spectra_viewer.py` | Tidy spectra plots |
| Peak Discovery | `tabs/peak_discovery.py` | Peak window diagnostics |
| Peak Feature Extraction | `tabs/peak_feature_extraction.py` | Targeted/dynamic peak tooling |
| Feature Analysis | `tabs/feature_analysis.py` | Distributions / exploratory stats |
| Sensor QC (legacy) | `tabs/sensor_qc_legacy.py` | CV-style QC + PDF |
| Sensor assessment | `tabs/sensor_assessment.py` | Regression QA, overlays, sensor-assessment QA PDF |
| Serotype Classification | `tabs/serotype_classification.py` | Serotype ML + classification report PDF |
| Regression V1: Global | `tabs/regression_global.py` | Global concentration regressors |
| Regression V2: Two-Stage | `tabs/regression_two_stage.py` | Two-stage paradigm |
| Regression V3: MTL | `tabs/regression_mtl.py` | Multi-task / spectral-net style paradigm |

Supporting: `tabs/regression_common.py` for shared regression UI glue.

---

## Appendix B — Prep mode (`apps/txt_to_excel.py`)

| Concern | Detail |
| --- | --- |
| Entry | Sidebar **Convert TXT → Excel** → `enter_prep_mode()` sets `_app_mode = "prep"` |
| Exit | Prep sidebar **← Back to Analysis** → `enter_analysis_mode()` |
| Empty state | Title **Raman TXT to Excel Merger** + info callout + numbered **Next steps** (mirrors analysis-mode onboarding) |
| Metadata UI | Sixteen fields in a five-column grid (`METADATA_FIELD_SPECS`); required vs optional marked in labels; Notes (field 16) optional |
| Workbook layout | 16 metadata rows → File Name / Special Treatment (per signal) → Target Concentration → Actual Concentration → spectral block |
| Per-signal ordering | Uploaded `.txt` files sorted by CFU token extracted from filename (`_extract_cfu_sort_key`) before row/column assembly |
| Concentration policy | Actual values required per signal; Target optional (blank → em dash `—` in merged preview legend: `Target: — / Actual: {n}`) |
| Metadata presets | JSON import/export (`SERS_metadata_preset.json`, `TEMPLATE_VERSION = 1`); selective per-field export checkboxes |
| Reload persistence | Prep **Reload Data** clears uploads and run-specific widgets (`sensor_id`, `test_id`, `connection_id`, `serotype`, `testing_time`, `notes`) but restores instrument geometry, acquisition settings, date, operator, and rinsate from `_txt2excel_persistent_metadata` |
| Preview / export isolation | Post-convert download + merged spectrum preview live in `@st.fragment` (`_render_prep_export_and_preview`) to limit widget-layout jitter |
| Library coupling | Uses `visualization.plot_spectra` + `components.shared_ui.render_figure_stretch`; workbook row builders are test-importable from `tests/` |

---

## Appendix C — Session state and keys (representative)

Peak artifacts are written via `state.write_peak_artifacts_to_state`; filter widgets use canonical column keys from `state.get_filter_widget_key`. PDF bytes typically live under keys passed to `render_pdf_download_section`. Prep mode uses prefixed keys (`txt2excel_*`, `_txt2excel_*`) documented in `txt_to_excel.py`. For definitive key names, refer to `apps/state.py`, `apps/txt_to_excel.py`, and call sites — avoid duplicating string literals in new tabs.

---

*End of audit document.*
