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

---

## 3. Architectural map (concise)

### 3.1 Package layout

| Area | Role |
| --- | --- |
| `src/sensd_sers_analysis/` | Library: I/O, preprocessing, features, assessment, classification, **regression paradigms**, visualization, PDF reports, application services |
| `apps/` | Streamlit “SERS Data Explorer”: upload, sidebar filters, **ten** analysis tabs |

**Import boundary:** `apps/` imports `sensd_sers_analysis.*`, `cache`, `components`, `tabs`, `theme`, `state`. The library does not import `apps/`.

### 3.2 `apps/app.py` (composition root)

Orchestrates: upload → `load_from_uploaded` / `LoadedDataBundle` → Raman sidebar bounds → `build_cached_derived_bundle` → `write_peak_artifacts_to_state` → filter catalog + `apply_cached_filters` → optional `merge_targeted_peaks_into_filtered_bundle` for downstream tabs → tab dispatch.

Compared to legacy audits, **`app.py` is materially thinner**: derivation and filtering are delegated to `application/` + `cache.py`.

### 3.3 Notable library surfaces

- **`data/io.py`:** Embedded-metadata Excel → wide/tidy; `RS_COL_PREFIX`, `META_COLS`, `load_sers_data`, `wide_to_tidy`, etc.
- **`processing/`:** Metadata enrichment, trim, filters, PCA, dynamic peaks (`peak_features.py`), **targeted peaks** (`targeted_peak_features.py`).
- **`assessment/`:** Consistency, outliers, degradation, batch variance, regression-based sensor QA (`sensor_assessment_regression.py`).
- **`classification/`:** Clean-frame prep for serotype ML, RF/SVM training, plots.
- **`regression/`:** Alternative concentration-regression paradigms (global, two-stage, multi-task / `MtlSpectralNet`), splits, metrics, plots — exercised from Streamlit tabs `regression_global`, `regression_two_stage`, `regression_mtl`.
- **`application/`:** Services bridging UI and domain (`*_service.py`), `contracts.py` DTOs, `dataset_pipeline.py`, `merge_targeted_peaks_into_filtered_bundle`, regression PDF builders via `regression_service.py`.
- **`visualization/`:** Spectra, stats, assessment plots, peak discovery plots, targeted peak plots.
- **`report/pdf_builder.py`:** Sensor assessment PDFs, sensor-assessment QA PDF, serotype classification report PDF, **regression** PDF helpers consumed by application services.

---

## 4. Primary data flows (end-to-end)

1. **Upload → wide/tidy:** Bytes → temp paths → `load_uploaded_bundle` / `load_sers_data_as_wide_and_tidy`.
2. **Derived bundle:** Preprocess → trim Raman → basic features + dynamic peaks → `DerivedDataBundle` + `PeakArtifacts`.
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

Caching covers major dataframe/service stages; plot construction and some tab bodies still rerun with Streamlit’s execution model. Further gains would come from `st.fragment`, deferring expensive actions behind buttons, or artifact caching keyed more aggressively — without changing scientific outputs.

### 5.6 Package-level hygiene

- Root `__init__.py` exposes a **small** curated API (data, assessment summaries, core plots, one PDF builder); the app correctly imports `processing`, `classification`, `application`, etc. via subpackages.
- **`ConsistencyResult` / `DegradationResult`:** Defined dataclasses are not always the outward API; either adopt them as return types or simplify exports over time.

### 5.7 Documentation drift

Single source for “what exists” should remain aligned with `THEORY_AND_IMPLEMENTATION.md` (modules, tabs, config). This audit should be updated when adding new tabs (e.g. regression), services, or cross-cutting concerns.

---

## 6. Refactoring backlog (prioritized, behavior-preserving)

1. **Tighten PDF error boundaries** in `shared_ui` (specific exceptions + structured logging).
2. **Reduce visualization→assessment imports** by threading precomputed results into assessment plots.
3. **Namespace session reset** keys under a single prefix for safer reload behavior.
4. **Consolidate duplicated pooling/IQR orchestration** between plotting and regression QA when next refactoring those files.
5. **Extend characterization tests** beyond application services as critical paths grow (`regression/`, targeted peaks).

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

## Appendix B — Session state and keys (representative)

Peak artifacts are written via `state.write_peak_artifacts_to_state`; filter widgets use canonical column keys from `state.get_filter_widget_key`. PDF bytes typically live under keys passed to `render_pdf_download_section`. For definitive key names, refer to `apps/state.py` and call sites — avoid duplicating string literals in new tabs.

---

*End of audit document.*
