# SENSD SERS Analysis — Practical Theory & Implementation Guide

Practical reference for **what this repository currently does**: research context, data model, the processing pipeline from raw Excel ingestion through statistical quality assurance to machine-learning classification, the Streamlit application layer, and the main modules in the package. Dependencies and tool versions live in **`pyproject.toml`** (Python ≥ 3.12; NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn, openpyxl, Streamlit, ReportLab).

This file is a practical reference. Resolve disagreements between scientific requirements and current behavior explicitly; neither code nor documentation alone establishes scientific correctness.

**Keeping this file current:** When a change affects public behavior, module boundaries, data flow, or how someone runs the application, update this document in the same change as the code.

**Documentation map**

- **`docs/THEORY_AND_IMPLEMENTATION.md` (this file):** Strategy, domain context, processing pipeline theory, statistical QA foundations, ML classification details, architecture layers, and module reference (what each part *does*).
- **`docs/ARCHITECTURE_AUDIT_SRC_AND_APPS.md`:** Structural audit of the `src/` vs `apps/` boundary, layering, remaining technical debt, and refactor backlog (how the code is *organized*). It complements this file and is **not** a substitute for it.
- **`AGENTS.md`:** AI agent rules, coding standards, and project conventions.
- **`README.md`:** Installation and quick-start instructions.

---

## Table of contents

1. [Project philosophy & research context](#1-project-philosophy--research-context)
2. [Domain primer: SERS for pathogen detection](#2-domain-primer-sers-for-pathogen-detection)
3. [Data origin, format & naming conventions](#3-data-origin-format--naming-conventions)
4. [Architecture layers and dependency direction](#4-architecture-layers-and-dependency-direction)
5. [Data processing pipeline overview](#5-data-processing-pipeline-overview)
6. [Stage 1: Raw data ingestion & metadata normalization](#6-stage-1-raw-data-ingestion--metadata-normalization)
7. [Stage 2: Spectral alignment & feature extraction](#7-stage-2-spectral-alignment--feature-extraction)
8. [Stage 3: Dynamic peak detection](#8-stage-3-dynamic-peak-detection)
9. [Stage 4: Outlier detection & statistical QA](#9-stage-4-outlier-detection--statistical-qa)
10. [Stage 5: Sensor consistency & degradation analysis](#10-stage-5-sensor-consistency--degradation-analysis)
11. [Stage 6: Model-based sensor QA (regression pipeline)](#11-stage-6-model-based-sensor-qa-regression-pipeline)
12. [Stage 7: Batch-level assessment & multi-sensor exclusion](#12-stage-7-batch-level-assessment--multi-sensor-exclusion)
13. [Stage 8: ML classification (serotyping)](#13-stage-8-ml-classification-serotyping)
14. [Application layer: Streamlit frontend](#14-application-layer-streamlit-frontend)
15. [Package map: `sensd_sers_analysis`](#15-package-map-sensd_sers_analysis)
16. [Module reference: `data/`](#16-module-reference-data)
17. [Module reference: `processing/`](#17-module-reference-processing)
18. [Module reference: `assessment/`](#18-module-reference-assessment)
19. [Module reference: `classification/`](#19-module-reference-classification)
20. [Module reference: `config/`](#20-module-reference-config)
21. [Module reference: `application/`](#21-module-reference-application)
22. [Module reference: `visualization/`](#22-module-reference-visualization)
23. [Module reference: `report/`](#23-module-reference-report)
24. [Module reference: `utils/`](#24-module-reference-utils)
25. [Streamlit app: `apps/`](#25-streamlit-app-apps)
26. [Tests](#26-tests)
27. [Policy constants & configuration SSOT](#27-policy-constants--configuration-ssot)
28. [Design choices, limitations & writing-up guidance](#28-design-choices-limitations--writing-up-guidance)
29. [Known gaps & optional extensions](#29-known-gaps--optional-extensions)

---

## 1. Project philosophy & research context

### 1.1 The problem in one sentence

We are building a **computational pipeline** that ingests raw Surface-Enhanced Raman Spectroscopy (SERS) sensor data, provides retrospective quality diagnostics and trains independently evaluated baseline machine-learning classifiers to distinguish *Salmonella* serotypes from negative controls.

### 1.2 Why this pipeline exists

SERS biosensors promise rapid, label-free detection of bacterial pathogens at low concentrations. However, the raw spectral data from these sensors is noisy, variable across sensor units and test sessions, and complicated by background matrix effects (e.g., turkey rinsate). A researcher analyzing this data faces three interleaved problems:

1. **Data heterogeneity:** Files arrive from different sensor units, test dates, operators, and concentrations. Metadata is embedded inside Excel files, not in filenames or separate manifests.
2. **Quality assurance:** Not all sensors behave consistently. Some degrade over repeated use; others produce flat or erratic responses. These can be studied retrospectively without making evaluated model eligibility depend on held-out responses.
3. **Serotype discrimination:** For explicitly identified samples, the question is whether SERS spectral features can distinguish *Salmonella* Typhimurium (ST) from *Salmonella* Enteritidis (SE) and from explicit rinsate controls.

This codebase addresses all three problems in a single, modular pipeline that is operable both programmatically (as a Python package) and interactively (via a Streamlit web application).

### 1.3 Scope and non-goals

**In scope:** Instrument `.txt` → embedded Excel prep (Streamlit), data loading and parsing, metadata normalization, master-grid Raman alignment, spectral trimming, scalar and peak-based feature extraction, PCA dimensionality reduction, IQR/z-score outlier detection, regression-based sensor consistency assessment, batch-level sensor exclusion, degradation trend analysis, baseline Random Forest / SVM classification, and concentration-regression paradigms (global, two-stage, MTL).

**Not implemented:** Additional spectral baseline correction, cosmic ray removal, spectral deconvolution, real-time inference, hardware diagnostics, and cross-laboratory reproducibility studies. The existing MTL model is a small neural network; further model complexity depends on demonstrated needs.

---

## 2. Domain primer: SERS for pathogen detection

This section provides only the domain context necessary to understand the codebase's design decisions. It is not a comprehensive review of SERS physics.

### 2.1 What the sensor measures

A SERS sensor illuminates a sample with a laser. Molecules adsorbed on nanostructured metallic surfaces produce **Raman-scattered photons** at wavelengths shifted from the excitation laser by amounts characteristic of their vibrational modes. The instrument records **intensity as a function of Raman shift** (in cm⁻¹), producing a spectrum. Each spectrum is a 1D signal: an array of intensities indexed by wavenumber.

### 2.2 The experimental setup (as reflected in the data)

From the data README files and the codebase metadata schema:

- **Sensors:** Multiple sensor units (identified by `sensor_id`) of the same design.
- **Serotypes:** Primarily *Salmonella* Typhimurium (ST) and *Salmonella* Enteritidis (SE). Some datasets include mixed serotypes.
- **Concentrations:** Serial dilutions from 1000 CFU down to 0 CFU (turkey rinsate negative control). Concentrations are recorded per-signal within each Excel file.
- **Replicates:** Multiple signals (spectra) per test, captured as columns within a single Excel file.
- **Temporal axis:** Multiple test sessions (`test_id`) per sensor, enabling degradation analysis over time.

### 2.3 What makes this data challenging

- **Sensor-to-sensor variability:** Identical experimental conditions can produce different absolute intensities across sensor units.
- **Background interference:** 0 CFU (rinsate) samples contain matrix effects that can mimic or mask pathogen-specific peaks.
- **Low-concentration regime:** At 1 CFU, signal-to-noise ratios are low. Peaks may not be detectable.
- **Sensor degradation:** Repeated use can degrade SERS substrates, causing systematic signal decline.

---

## 3. Data origin, format & naming conventions

### 3.1 The embedded Excel format

Each SERS data file is a self-contained Excel workbook (`.xlsx`) with the following structure, parsed by `data/io.py::_parse_embedded_format`:

```text
Row 1–16:  File-level metadata (Sensor ID, Test ID, geometry, date, …)
Row 17:    File Name           | heat_kill.txt    live_1000.txt    ...
Row 18:    Special Treatment   | heat kill        (blank)          ...
Row 19:    Target Concentration| (blank)          1000             ...
Row 20:    Actual Concentration| 500              1000             ...
Row 21:    Raman Shift         | Relative Light intensity (a.u) × N signals
Row 22+:   400.00              | 1234.5           1245.6           ...
...        (Raman shift)        (intensity per signal)
```

**File-level metadata block** (rows 1–16 in prep-generated workbooks): Key-value pairs with normalized keys (`sensor id`, `test id`, `connection id`, `serotype`, sensor geometry, date, operator, …). Legacy collaborator files may use a shorter block (rows 0–3) with only the four required IDs plus serotype.

Prep-generated workbooks use sixteen fixed metadata rows (disk geometry, acquisition settings, sensor/test IDs, serotype, rinsate, date/time, operator, notes). Numeric validation applies to geometry and acquisition fields (1–4, 6–7); date and time fields (13–14) use dedicated parsers; Notes (16) is optional.

**Per-signal label rows** (optional in legacy files): **File Name** maps to `source_txt_filename` (instrument `.txt` provenance). **Special Treatment** maps to `special_treatment` (e.g., heat kill, PAA). The parser accepts these rows either above or below the concentration block.

**Concentration rows:** When an **Actual Concentration** row exists, numeric concentrations for analysis are read from that row; otherwise the first row whose column-A label contains `"concentration"` is used (legacy single-row layout). Target Concentration may be blank per signal (prep mode); only columns with valid numeric actual concentrations become signal columns.

**Data block** (first row where column 0 is numeric): Column 0 = Raman shift values (cm⁻¹). Columns 1..N = intensity values per signal. Each column is one spectrum from the same test/sensor/serotype combination.

### 3.2 Wide-format DataFrame convention

After parsing, each file becomes rows in a **wide-format** DataFrame:

| Column type | Examples | Description |
|-------------|----------|-------------|
| Metadata | `sensor_id`, `test_id`, `serotype`, `concentration`, `filename`, `signal_index`, `source_txt_filename`, `special_treatment` | One row per signal |
| Spectral | `rs_400.00`, `rs_401.00`, ... `rs_1800.00` | Intensity at each Raman shift |

The `rs_` prefix and 2-decimal rounding are constants defined in `data/io.py` (`RS_COL_PREFIX`, `RAMAN_SHIFT_DECIMALS`). This naming convention enables programmatic column selection: any column starting with `rs_` is a spectral intensity value.

### 3.3 Tidy-format DataFrame

For plotting, the wide format is melted into a **tidy** (long) format via `wide_to_tidy`:

| Column | Description |
|--------|-------------|
| All `META_COLS` present | Metadata carried through |
| `raman_shift` | Float, extracted from column name |
| `intensity` | Float, the measured value |

### 3.4 Metadata columns (`META_COLS`)

The canonical metadata column list, defined in `data/io.py`:

```python
META_COLS = [
    "sensor_model", "sensor_id", "test_id", "connection_id",
    "serotype", "date", "operator", "concentration",
    "filename", "source_txt_filename", "special_treatment",
    "signal_index",
]
```

`filename` and `signal_index` together form a unique identifier for each spectrum trace. `source_txt_filename` and `special_treatment` are populated when workbooks include the prep-tool per-signal rows; legacy files load with empty strings for those columns.

---

## 4. Architecture layers and dependency direction

```text
data  →  processing  →  assessment  ┬→  classification  →  report
(I/O)    (features)     (QA/stats)    │    (ML models)       (PDF)
                                     └→  regression ────────┘
                                         (concentration regression paradigms)
                    ↘                ↗
                      config (policy constants)
                    ↗                ↘
              utils                    visualization
              (labels, sort, parse)    (matplotlib/seaborn plots)

              ─── application ───
              (service layer: bridges src/ and apps/)

              ─── apps/ ───
              (Streamlit UI, caching, session state)
```

**Dependency rule:** Lower layers do not import higher layers. `data` knows nothing about `assessment`. `processing` knows nothing about `classification`. The `application/` layer is the integration point that wires services together for the Streamlit frontend.

**`config/`** is a leaf dependency: it exports only constants and is imported by `assessment`, `classification`, and `application`.

---

## 5. Data processing pipeline overview

The processing and analysis views below cover ingestion, exploration, diagnostics, and modeling. QA and model evaluation are separate branches; diagnostic exclusions do not determine default model eligibility. Each stage maps to specific modules in `src/sensd_sers_analysis/`.

```text
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Raw Data Ingestion & Metadata Normalization           │
│  data/io.py → processing/metadata.py                            │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Spectral Alignment & Scalar Feature Extraction        │
│  processing/alignment.py (snap + trim) → processing/features.py │
│  processing/pca_features.py                                     │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Dynamic Peak Detection                                │
│  processing/peak_features.py                                    │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: Outlier Detection (IQR / Z-score)                     │
│  assessment/outliers.py                                         │
├─────────────────────────────────────────────────────────────────┤
│  Stage 5: Sensor Consistency & Degradation                      │
│  assessment/consistency.py → assessment/degradation.py           │
├─────────────────────────────────────────────────────────────────┤
│  Stage 6: Model-Based Sensor QA (Two-Pass Regression)           │
│  assessment/sensor_assessment_regression.py                                │
├─────────────────────────────────────────────────────────────────┤
│  Stage 7: Batch-Level Multi-Sensor Exclusion                    │
│  assessment/sensor_assessment_regression.py → assessment/batch_variance.py  │
├─────────────────────────────────────────────────────────────────┤
│  Stage 8: ML Classification (Serotyping)                        │
│  classification/data_prep.py → classification/models.py          │
└─────────────────────────────────────────────────────────────────┘
```

The orchestration of these stages is performed by `application/dataset_pipeline.py` (Stages 1–3) and the individual service modules in `application/` (diagnostics and models). **Concentration regression** experiments (global, two-stage, multi-task spectral models) live in `regression/` and are wired through `application/regression_service.py` and the Streamlit regression tabs — they sit alongside classification as an alternate modeling track, not as an extra numbered “Stage” in the QA pipeline above.

---

## 6. Stage 1: Raw data ingestion & metadata normalization

### 6.1 File parsing (`data/io.py`)

The entry point is `load_sers_data(paths)`, which accepts files, folders, or a mix. Internally:

1. **`_collect_files`** resolves paths to a flat list of `.xlsx` files, skipping files prefixed with `~` or `_` (temp/hidden files).
2. **`_parse_embedded_format`** reads each workbook headerlessly (`header=None`), locates concentration rows by scanning column 0 for `"concentration"`, prefers **Actual Concentration** when present, extracts the metadata block above the first concentration row (skipping per-signal label rows), parses optional **File Name** / **Special Treatment** rows, and reads the signal data below.
3. **`_load_signal_file`** assembles metadata + transposed signal matrix into a wide DataFrame. Raman shift values are rounded to `RAMAN_SHIFT_DECIMALS = 2` decimal places and prefixed with `rs_`. Per-signal provenance lists are padded or trimmed to the signal count.
4. Non-finite spectral coordinates/intensities, duplicate rounded Raman labels, and metadata names colliding with spectral/computed fields are reported as per-file load failures. Within the parsed spectral block, rows containing intensity data require a coordinate; fully blank separator rows contain no measurement and may be ignored. Wide-to-tidy conversion retains all non-spectral metadata.
5. Files are concatenated with `ignore_index=True`. If a `serotypes` filter is provided, files whose embedded serotype does not match are skipped.

**Design choice:** Embedding metadata inside Excel files (rather than in filenames or a separate manifest) is driven by the experimental workflow—collaborators produce self-contained files from the instrument software. The parser is tolerant of optional fields (`date`, `operator`, `sensor_model`) but strict about required fields (`sensor id`, `test id`, `connection id`, `serotype`).

### 6.2 Metadata preprocessing (`processing/metadata.py`)

After loading, `preprocess_metadata(df)` enriches the DataFrame with derived columns:

**Log concentration:** `add_log_concentration` computes `log10(concentration)` for all positive concentrations. Zero concentrations (0 CFU) produce `NaN` because `log10(0)` is undefined. This is intentional—0 CFU samples are excluded from regression-based analyses that use `log_concentration` as the independent variable.

**Concentration grouping:** `add_target_concentration_group` labels each recorded initial target exactly, with categories in natural order. Missing targets remain `Unknown`; actual CFU values never supply nominal labels. `add_concentration_group` provides the same labels under the compatibility key `concentration_group`. Explicit Sample Type determines bacterial/control identity independently of target and actual concentration.

**Date normalization:** Dates are parsed to `datetime` and formatted as `YYYY-MM-DD` strings.

### 6.3 Theoretical basis

The log-transform of concentration is standard practice in dose-response analysis. For serial dilution experiments, concentration values span orders of magnitude (0–1000 CFU). A log-scale transform makes the spacing uniform and enables linear regression of `feature ~ log10(concentration)` (§11). The choice of `log10` (rather than natural log) aligns with the conventional CFU notation.

---

## 7. Stage 2: Spectral alignment & feature extraction

### 7.1 Master grid alignment (`processing/alignment.py`)

When multiple Excel files (or sensors) are loaded, native Raman-shift grids may differ in span, density, or numeric labels. Before trimming or feature extraction, `build_derived_bundle` calls `snap_spectra_to_master_grid`:

1. **Master selection:** Choose the spectrum row with the largest native spectral span; ties break on point density (points per cm⁻¹), then point count.
2. **Per-row interpolation:** Linearly interpolate each row onto master shifts that fall strictly within that row's native `[min_shift, max_shift]` support. Master coordinates outside the overlap remain **NaN** (no extrapolation).
3. **Near-duplicate merge:** On each row, consecutive shifts within `numpy.isclose` tolerances (`SNAP_SHIFT_DEDUPE_RTOL`, `SNAP_SHIFT_DEDUPE_ATOL` from `config/spectral_policies.py`) are merged via `nanmean` before interpolation.

**Design choice:** Snapping enables vectorized feature extraction and PCA across heterogeneous uploads without forcing a fixed instrument grid at parse time. NaN outside overlap preserves honest missing-data semantics when sensors do not share a wavenumber window.

### 7.2 Raman shift trimming (`processing/alignment.py`)

`trim_raman_shift(wide_df, min_shift, max_shift)` drops `rs_*` columns whose wavenumber falls outside the user-specified window. This serves two purposes:

1. **Noise reduction:** Spectral endpoints often have higher noise due to detector edge effects.
2. **Cross-sensor alignment:** Different sensor configurations may cover slightly different Raman shift ranges. Trimming to a common window prevents features from being extracted in non-overlapping regions.

### 7.3 Scalar feature extraction (`processing/features.py`)

`extract_basic_features(df_wide)` computes three macro-level scalar features per spectrum:

**Max intensity:**

```text
max_intensity = max(I(ν))  over all ν in the spectral window
```

Implemented as `np.nanmax(signals, axis=1)`. Provides a crude measure of overall signal strength. Sensitive to single-point spikes but robust to baseline offset.

**Mean intensity:**

```text
mean_intensity = (1/N) × Σ I(νᵢ)  for i = 1..N
```

Implemented as `np.nanmean(signals, axis=1)`. Averages over the full spectral window. Less sensitive to single-point noise than max_intensity but affected by baseline level.

**Integral area (trapezoidal integration):**

```text
integral_area ≈ Σ (I(νᵢ) + I(νᵢ₊₁)) / 2 × (νᵢ₊₁ - νᵢ)  for i = 1..N-1
```

Implemented via `scipy.integrate.trapezoid(signals, x=raman_shift, axis=1)`. The trapezoidal rule approximates the area under the spectrum curve using the actual Raman shift values as the x-axis. This accounts for potentially non-uniform spacing in the wavenumber grid. Non-finite results or fewer than two coordinates produce an unavailable area (`NaN`) and a warning. No intensity sum or imputation substitutes for an integral, and unavailable rows do not change other rows.

**Design choice:** These three features were chosen for their robustness to noise. They do not require peak identification or baseline subtraction—important when working with noisy SERS data where peak detection may fail for low-concentration samples. They serve as the primary features for sensor consistency assessment (§10–§12) and as base features for classification (§13).

### 7.4 PCA feature extraction (`processing/pca_features.py`)

`add_pca_features(df_wide, n_components=2)` is an exploratory full-cohort transformation. It fits StandardScaler and deterministic full-SVD PCA on complete finite spectra only, then aligns PC1/PC2 scores and explained-variance ratios to every source row. Incomplete spectra keep unavailable scores. Fewer than two complete rows leaves PCA unavailable; constant spectra have zero scores and undefined variance ratios. No zero imputation occurs.

PCA describes orthogonal directions of observed standardized variance. It does not establish serotype specificity or an independent source of physical variance. These cohort scores, their availability masks and dynamic discovery outputs do not supply default model predictors or decide model eligibility.

---

## 8. Stage 3: Dynamic peak detection

### 8.1 Overview and design rationale (`processing/peak_features.py`)

Peak detection is **serotype-specific** and **background-exclusive**: anchors (peak positions) and search windows use explicitly identified bacteria samples of each serotype, with recorded initial-target preferences. Rinsate controls do not define anchors. Only complete spectra contribute to the mean; no complete spectrum means no available anchors. Non-finite coordinates prevent peak discovery, and non-finite extraction windows remain unavailable. This labeled cohort recipe supports exploration and does not supply default ML predictors.

The pipeline produces `Peak_1_Height`, `Peak_2_Height`, ..., `Peak_K_Height` columns, where K is configurable per serotype via the Streamlit sidebar.

### 8.2 Smoothing: Savitzky-Golay filter

Before peak finding, the mean spectrum is smoothed using a **Savitzky-Golay filter** (`_smooth_spectrum`):

```text
ŷ[i] = SavGol(y, window_length=11, polyorder=3)
```

The Savitzky-Golay filter fits a polynomial of degree `polyorder` to successive windows of `window_length` points, replacing each central value with the polynomial's value at that point. This preserves peak shapes better than moving-average smoothing while suppressing high-frequency noise.

**Implementation guards:** If the spectrum has fewer than `window_length` points, smoothing is skipped. `window_length` is forced to odd. `polyorder` is clamped to `window_length - 1`.

### 8.3 Anchor discovery: `scipy.signal.find_peaks`

`_find_peaks_on_spectrum(x, y, n_peaks)` finds the top N most prominent peaks:

1. Apply Savitzky-Golay smoothing (§8.2).
2. Compute `prominence = max(0.01 × (max(y) - min(y)), 1e-10)`.
3. Compute `distance = min(15, len(x) // (n_peaks × 3))`.
4. Call `scipy.signal.find_peaks(y, prominence=prominence, distance=distance)`.
5. Sort by prominence descending; return the top `n_peaks` indices, sorted by position.

**Parameter choices as implemented:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `ANCHOR_PROMINENCE_FRAC` | 0.01 (1%) | Permissive threshold so biological peaks are not suppressed by baseline drift |
| `ANCHOR_DISTANCE_INDICES` | 15 | Minimum separation between peaks in index space; prevents detecting the same broad peak twice |

**Theoretical basis:** `scipy.signal.find_peaks` identifies local maxima. **Prominence** measures how much a peak stands out relative to surrounding valleys—it is the vertical distance from the peak to the higher of the two baseline points obtained by extending horizontally from the peak until the signal rises to the peak height. Using 1% of the total range as the prominence threshold is deliberately permissive: the top-N selection by prominence then picks the most salient peaks without requiring a hard absolute threshold.

### 8.4 Window boundary computation

For each serotype, `_compute_peak_windows_for_serotype` computes search windows around anchors using the high-concentration mean spectrum:

1. **Mean spectrum:** Computed from the highest-concentration subset (preference order: 1000, 100, 10, 1 CFU) using explicitly identified bacterial samples. Uses `np.nanmean` across all spectra in that subset.

2. **Inner boundaries (between adjacent anchors):** For anchors A_i and A_{i+1}, the boundary is placed at the **argmin of the mean spectrum** in `[A_i, A_{i+1}]`. This finds the true valley between peaks, robust to jagged slopes.

3. **Outer boundaries (leftmost and rightmost):** From the first anchor, search left until the mean spectrum drops to ≤ 5% of the peak height (`OUTER_BASELINE_FRAC = 0.05`). From the last anchor, search right with the same criterion.

```text
Inner boundary: B[i] = argmin(mean_spec[A_i : A_{i+1}])
Outer left:     search left from A_1 until mean_spec ≤ 0.05 × mean_spec[A_1]
Outer right:    search right from A_K until mean_spec ≤ 0.05 × mean_spec[A_K]
```

### 8.5 Peak height extraction

For each spectrum (row), peak heights are extracted using the serotype-specific windows:

1. Identify the window `[w_min, w_max]` for peak k.
2. Extract the intensity values within the window.
3. Compute a local baseline as the mean of the first and last 10% of window points.
4. Peak height = `max(window_intensities) - baseline`.
5. If peak height < `noise_threshold_frac × global_max_intensity` (default 2%), mark as NaN (peak not detected).

**Control handling:** Explicit rinsate controls use the default serotype's windows, chosen lexicographically from serotypes with discovered peaks. A measured zero does not make a bacterial sample a rinsate control.

### 8.6 Success rate

For each peak window, the `success_rate` is computed as the fraction of spectra (within that serotype) where a valid peak height was extracted (non-NaN). This diagnostic metric appears in the Peak Diagnostics tab of the Streamlit app and helps researchers identify peaks that are unreliable at low concentrations.

---

## 9. Stage 4: Outlier detection & statistical QA

### 9.1 IQR-based outlier detection (`assessment/outliers.py`)

`detect_outliers_iqr(values, whis=1.5)` implements the standard interquartile range method:

```text
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower fence = Q1 - whis × IQR
Upper fence = Q3 + whis × IQR
Outlier ⟺ value < Lower fence  OR  value > Upper fence
```

With the default `whis = 1.5`, this matches the standard boxplot whisker rule. For a normal distribution, approximately 0.7% of points fall outside the fences. The method is non-parametric and robust to non-normal distributions common in spectral data.

**Edge cases:** If `IQR = 0` (all values identical or nearly so), no outliers are flagged. NaN values are not counted as outliers (masked as `False`).

### 9.2 Z-score outlier detection

`detect_outliers_zscore(values, threshold=3.0)` uses the parametric approach:

```text
z[i] = |value[i] - mean| / std
Outlier ⟺ z[i] > threshold
```

Uses sample mean and standard deviation (not median/MAD, despite the docstring mentioning robust statistics—the implementation uses `np.mean` and `np.std`). A threshold of 3.0 corresponds to approximately 0.3% of points for normally distributed data.

### 9.3 Filter dispatch

`filter_outliers(df, feature_col, method="iqr")` dispatches to either method and returns `(inliers_df, outliers_df)`. This function is the building block for all downstream outlier-aware analyses.

---

## 10. Stage 5: Sensor consistency & degradation analysis

### 10.1 Coefficient of variation (`assessment/consistency.py`)

The **Coefficient of Variation** (CV) quantifies measurement repeatability:

```text
CV = σ / |μ|
```

where σ is the standard deviation and μ is the mean of replicate measurements. CV is dimensionless and enables comparison across features with different scales. Implemented in `coefficient_of_variation`; returns NaN if μ = 0.

### 10.2 Consistency metrics

`compute_consistency_metrics(df, feature_col, group_cols, outlier_method)` computes both raw and outlier-filtered statistics per group:

1. For each group (e.g., per `sensor_id × serotype × concentration_group`):
   - Compute raw: `mean_raw`, `std_raw`, `cv_raw` on all points.
   - Run outlier detection (IQR or z-score) → split into inliers and outliers.
   - Compute filtered: `mean_filtered`, `std_filtered`, `cv_filtered` on inliers only.
2. Report `n_total`, `n_inliers`, `n_outliers`.

The side-by-side raw vs. filtered CV reveals how much a few extreme points inflate apparent variability.

### 10.3 Degradation trend analysis (`assessment/degradation.py`)

`compute_degradation(df, feature_col, sequence_col, group_cols)` fits a simple linear regression per sensor to detect temporal trends:

```text
feature = slope × sequence + intercept
```

where `sequence` is the test ordinal (1, 2, 3, ...) derived from `test_id` or `date` ordering.

**Slope interpretation** (as implemented):

| Condition | Interpretation |
|-----------|---------------|
| `|slope / mean(y)| × 100 < 0.5%` | **Stable** |
| `slope < 0` | **Degradation** (feature decreasing over time) |
| `slope > 0` | **Improvement** |

The 0.5% relative threshold prevents labeling negligible trends as significant. `scipy.stats.linregress` provides slope, intercept, R², p-value, and standard error.

**Data preparation:** `prepare_degradation_data` aggregates the feature to one value per `(sensor_id, test_id)` using the group mean, then assigns a `test_ordinal` per sensor ordered by date (or test_id if date is unavailable). This ensures the regression x-axis represents temporal order, not within-file signal order.

---

## 11. Stage 6: Model-based sensor QA (regression pipeline)

### 11.1 The core idea

Rather than assessing sensor quality by summary statistics alone, the model-based approach evaluates whether a sensor produces a **consistent dose-response relationship**. A well-functioning sensor should show a predictable relationship between pathogen concentration and spectral feature intensity. Sensors with poor fit (high RMSE or low R²) are likely malfunctioning or degraded.

### 11.2 Single-pass regression (`assessment/sensor_assessment_regression.py`)

`fit_concentration_regression(df, feature_col)` fits:

```text
feature = slope × log₁₀(concentration) + intercept
```

on all rows where `log_concentration` is not NaN (i.e., excludes 0 CFU). Uses `scipy.stats.linregress`. Returns `ConcentrationRegressionResult` with slope, intercept, R², RMSE, fitted values.

**RMSE** (Root Mean Square Error):

```text
RMSE = √( (1/N) × Σ (yᵢ - ŷᵢ)² )
```

RMSE is in the same units as the feature and directly quantifies the average prediction error.

### 11.3 Two-pass cleaned regression

`fit_concentration_regression_cleaned(df, feature_col)` implements a two-pass approach:

**Pass 1 (raw fit):** Fit all valid >0 CFU points. Compute residuals. Identify outliers via IQR on **absolute residuals**:

```text
|residual| > Q3(|residuals|) + whis × IQR(|residuals|)
```

This targets points that deviate more from the regression line than expected, not points that are extreme in the feature space. The default `whis = 1.5` matches `GLOBAL_QA_IQR_WHIS`.

**Pass 2 (clean fit):** Remove outlier points. Refit on inliers. Compute clean RMSE and R².

**Guard clauses:** Returns `None` if fewer than 2 valid points exist. If x has no variance (single concentration level), returns `None`. If fewer than 2 inliers remain after outlier removal, the clean fit falls back to the raw fit.

### 11.4 Zero-CFU baseline

`get_zero_cfu_baseline(df, feature_col)` computes the mean feature value for explicitly identified rinsate controls. It is a control-response reference, displayed as a horizontal line. Concentration alone does not identify controls or establish a noise floor.

---

## 12. Stage 7: Batch-level assessment & multi-sensor exclusion

### 12.1 Global QA pipeline (`assessment/sensor_assessment_regression.py`)

`get_global_model_consistency_qa(df, feature_cols)` applies the two-pass regression (§11.3) to every combination of `(sensor_id, serotype, feature)` and then applies a **dual-threshold exclusion rule**:

A sensor is marked **"Excluded"** for a given `(serotype, feature)` if:

```text
Condition A (too noisy):   Clean RMSE > 2.0 × batch_median(Clean RMSE)
    OR
Condition B (dead/flat):   Clean R² < 0.80
```

**Condition A** catches sensors with abnormally high regression error relative to the batch. The multiplier `GLOBAL_QA_REJECTION_MULTIPLIER = 2.0` means a sensor must have more than twice the typical batch error to be excluded.

**Condition B** catches unresponsive sensors that produce near-constant output regardless of concentration. An R² below 0.80 indicates that the linear model explains less than 80% of the variance in the data.

The function returns both the QA table and an `excluded_map: dict[(serotype, feature) → set[sensor_id]]` used by downstream stages.

### 12.2 Batch variance analysis (`assessment/batch_variance.py`)

`compute_batch_variance(df, feature_col, sensor_col)` provides a complementary view:

1. Per sensor: compute `n_samples`, `mean`, `std`, `CV`.
2. Across sensors: compute `batch_mean` and `batch_std` of per-sensor means.
3. Per sensor: compute `z_from_batch = (sensor_mean - batch_mean) / batch_std` and `deviation_pct`.

`identify_deviating_sensors(batch_df, z_threshold=2.0)` flags sensors where `|z_from_batch| > 2.0` (default `BATCH_DEVIATION_Z_THRESHOLD`).

This analysis complements the regression-based QA by detecting sensors that differ from the batch in absolute feature level, even if their dose-response curve shape is acceptable.

### 12.3 Macro batch regression

`compute_macro_batch_regression` pools inlier data from all **Pass** sensors (per serotype) and fits a single regression, providing a batch-level RMSE and R². This answers: "How consistent is the entire good-sensor batch?"

The pooling process:

1. For each Pass sensor, run the two-pass cleaned regression to identify inliers.
2. Pool all inlier `(log_concentration, feature)` pairs.
3. Fit a single regression on the pooled data.
4. Apply another round of IQR-based outlier detection on the pooled residuals for macro-level outlier identification.

`MacroRegressionResult` retains each pooled point's sensor ID alongside its coordinates and outlier mask. `plot_macro_batch_regression` uses this artifact directly; a supplied result does not trigger another per-sensor fit or pooling pass.

---

## 13. Stage 8: ML classification (serotyping)

### 13.1 Data preparation (`classification/data_prep.py`)

`label_classification_dataset` supplies the model-service cohort: explicit Rinsate control rows receive the Rinsate class; explicit bacterial rows need a usable serotype. It preserves source indices and order. No response-based QA exclusions or residual screening occurs. Unknown identities remain visible in Inventory but are ineligible for class models.

`prepare_classification_dataset` remains a retrospective diagnostic API. It applies the supplied sensor-exclusion map and pooled residual screening, then labels identities. Absence from an exclusion map does not establish a Pass assessment. `prepare_concentration_regression_data(..., apply_response_screening=True)` exposes that diagnostic cleaning for existing callers; application model services explicitly pass False.

Regression requires bacterial identity, finite positive measured actual concentration and a finite log10 target. Nominal targets cannot supply missing actual measurements. Controls remain in classification but receive no regression target. Custom class-column names cannot overwrite another input column.

### 13.2 Feature set

| Model predictor | Quantity |
| --- | --- |
| `integral_area` | Trapezoidal area on actual Raman coordinates; intensity × cm⁻¹ |
| `max_intensity` | Maximum spectral intensity |
| `mean_intensity` | Mean spectral intensity |
| `peak_near_*` | Per-spectrum baseline-adjusted maximum inside a fixed anchor window |

Cohort PC1/PC2, dynamic discovered peaks, metadata, labels, concentrations and identifiers are outside the default predictor list. Fixed anchors/windows must be chosen before evaluation; revising them after inspecting test outcomes preserves a retrospective limitation. A selected missing column, NaN or Inf fails explicitly with source/feature context. Finite measured zeros remain zero. Optional metadata and unselected exploratory NaNs do not block training; no selected feature or row is silently dropped or zero-imputed.

### 13.3 Model training (`classification/models.py`, `regression/`)

Classification and concentration models use an 80/20 sensor-group holdout with configurable test fraction/seed (default 0.2/42). Missing or blank group IDs fail. The dataframe classifier requires at least two training classes and rejects test classes without training support. The arrays API accepts caller-defined independent splits and optional training sensor IDs; callers own outer disjointness. Classification results and regression artifacts retain positional split definitions.

RF and RBF-SVM/SVR are baseline candidate families. RandomizedSearchCV uses a realized training-sensor GroupKFold plan and the same scorer for both candidates: weighted F1 for classification, negative log-concentration RMSE for regression. Each estimator owns a Pipeline with fold-local StandardScaler. After search, the chosen candidate refits on all outer training rows; prediction receives unscaled selected features exactly once. Search is skipped below the configured training-size threshold or when the predeclared fold support is infeasible. Independently supplied arrays without group IDs use stratified row CV.

Model-family selection uses comparable training CV scores only. Ties choose RF. When both candidates lack feasible CV, RF is the predeclared reference. Fit exceptions or non-finite scores abort; they never trigger a rescue fallback. Held-out accuracy, weighted precision/recall/F1 and RMSE/MAE/R² describe evaluation rather than selection. Tree impurity importance describes the fitted model, not causal value or peak specificity. This separation follows [scikit-learn's training-only preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html).

Two-stage regression selects its routing classifier by training CV and reports routed metrics across all held-out rows. Oracle routing uses the true serotype as a separate diagnostic; its realized RMSE is not a guaranteed bound. All conditional regressors exclude outer test sensors.

MTL derives its class vocabulary from outer training rows and rejects unseen held-out classes. Early stopping holds out sensor groups inside the outer training set (seed +17); inner training must support every outer training class. Its scaler fits inner training sensors only and stays frozen. The best weights are copied, restored, and fine-tuned with a fresh AdamW optimizer for a fixed `max(1, min(30, max_epochs // 4))` budget on all outer training rows. No further selection uses the former inner validation or outer test set. Returned histories describe early stopping before fine-tuning; returned inner indices and fine-tuning epochs record the protocol.

### 13.4 Theoretical basis

Random Forest is an **ensemble of decision trees** trained on bootstrap samples with random feature subsets at each split. It is robust to overfitting and provides feature importance scores. RBF-SVM maps inputs to a high-dimensional feature space via the kernel `K(x, x') = exp(-γ ||x - x'||²)` and finds the maximum-margin hyperplane. Both are well-established baselines for small-to-medium tabular classification tasks. The choice of these models (rather than deep learning) is appropriate given the dataset size (hundreds to low thousands of samples) and the interpretability requirements of a research setting.

---

### 13.5 Validation tables and sample alignment

`assessment/validation_metrics.py` uses fixed RF references without model-family selection and predicts with repeated sensor holdouts and computes identification rates only on held-out rows. Classification and regression frames retain unique original source indices. Regression predictions are reindexed onto classification rows; controls and other rows without regression predictions remain unavailable. Model arrays and split masks use row positions independently of source labels. Each round creates one classification sensor assignment; regression training rows are derived from those same held-in sensor IDs. A held-out sensor cannot enter either component. Rates average every requested successful round; failures abort the entire request.

A failed classifier or regressor fit raises a contextual exception including the fold, and the Validation UI displays an error without tables or downloads for that request. True labels never substitute for predictions. With fewer than two sensors, no model is fitted; counts/CV remain available, evaluation counts are zero, and ML rates are blank or N/A.

## 14. Application layer: Streamlit frontend

### 14.1 Architecture

The `apps/` directory contains a Streamlit application with **two modes** toggled by session state key `_app_mode`. Both modes share `st.set_page_config(page_title="SERS Data Explorer", layout="wide")` at the composition root; prep mode sets its own in-page title (**Raman TXT to Excel Merger**).

| Mode | Module | Purpose |
| --- | --- | --- |
| **Analysis** (default) | `app.py` + tabs | Embedded Excel upload, filtering, ten analysis tabs |
| **Prep** | `txt_to_excel.py` | Merge instrument `.txt` exports → embedded `.xlsx` for downstream upload |

**Empty-state onboarding:** When no files are loaded, each mode renders a title, an info callout, and a numbered **Next steps** checklist rather than a blank page. Analysis mode stops after the checklist until sidebar upload succeeds; prep mode returns early until `.txt` files appear in the sidebar.

Analysis mode uses a layered architecture:

```text
apps/app.py                     ← Entry point, mode switch, layout, tab dispatch
apps/txt_to_excel.py            ← TXT→Excel prep shell (invoked when _app_mode == "prep")
apps/cache.py                   ← @st.cache_data wrappers
apps/state.py                   ← Session state adapters
apps/theme.py                   ← UI constants (sizes, colors, HTML)
apps/components/                ← Reusable UI components
  data_loading.py               ← File upload, cache clearing
  filter_ui.py                  ← Dynamic metadata filter widgets
  raman_sidebar.py              ← Raman shift and peaks controls
  shared_ui.py                  ← PDF download, metric rows, figure rendering
apps/tabs/                      ← Tab-specific rendering
  spectra_viewer.py             ← Plot spectra with hue/style/variance
  peak_discovery.py             ← Peak discovery (mean spectrum + signal checks)
  peak_feature_extraction.py    ← Targeted anchor peak heights
  feature_analysis.py           ← Distributions, correlations, vs log concentration
  sensor_qc_legacy.py           ← Legacy CV-style sensor QC + PDF
  sensor_assessment.py          ← Regression QA, global assessment, overlay + PDF
  serotype_classification.py    ← PCA scatter, RF/SVM, confusion matrix
  regression_global.py          ← Concentration regression V1 (global regressors)
  regression_two_stage.py       ← Concentration regression V2 (two-stage)
  regression_mtl.py             ← Concentration regression V3 (MTL / spectral net)
  regression_common.py          ← Shared helpers for regression tabs
```

**Prep mode workflow:** Sidebar **Convert TXT → Excel** calls `enter_prep_mode()`. Users upload `.txt` files (sorted by CFU token in the filename for stable per-signal column order), complete sixteen metadata fields in a five-column grid (required fields marked `*`, Notes optional), optionally import/export a JSON metadata preset (`SERS_metadata_preset.json`, version 1, with selective field export), enter per-signal Target (optional) and Actual (required) concentrations plus optional Special Treatment labels, merge spectra within a Raman window, preview with `plot_spectra` (legend entries like `Target: — / Actual: 1000 (live_1000.txt)` when target is blank), and download an embedded workbook. Post-convert download and preview render inside `@st.fragment` to avoid rerunning metadata widgets. Prep **Reload Data** clears uploads and run-specific metadata (`sensor_id`, `test_id`, `connection_id`, `serotype`, testing time, notes) but persists instrument geometry, acquisition settings, date, operator, and rinsate via a session snapshot. **← Back to Analysis** returns to the main explorer; users then upload the saved `.xlsx` in analysis mode.

### 14.2 Data flow through the application

1. **Upload** → `apps/components/data_loading.py:load_from_uploaded` caches uploaded file bytes, delegates to `load_uploaded_bundle`, and returns `LoadedDataBundle(wide_df, tidy_df)`. Per-signal provenance columns are present when the workbook includes File Name / Special Treatment rows.
2. **Derived bundle** → `build_derived_bundle` applies `preprocess_metadata`, **`snap_spectra_to_master_grid`**, `trim_raman_shift`, `extract_basic_features`, `extract_dynamic_peak_features` → `DerivedDataBundle(wide_df, tidy_df, features_df, peak_df, peak_artifacts)`.
3. **Filtering** → Dynamic sidebar filters built from metadata columns. `build_filter_catalog` identifies filterable columns and splits them into main (5) and "more" groups. Each filter supports include/exclude mode. `apply_filters` produces a `FilteredBundle(filtered_tidy_df, filtered_features_df, n_unique_spectra)`.
4. **Targeted peaks (optional):** Anchor positions from Peak Feature Extraction (session state) are merged into the filtered feature frame via `merge_targeted_peaks_into_filtered_bundle` before tabs that need those columns (`feature_analysis`, legacy QC, sensor assessment, serotype classification, regression tabs).
5. **Tabs** consume filtered data. Most repeated app-level computations are wrapped in `@st.cache_data` decorators. Plot creation itself still happens at render time.
6. **PDF reports** are generated on demand via ReportLab, stored in `st.session_state`, and offered for download.

### 14.3 Caching strategy

Most repeated app-level computations are memoized via `@st.cache_data`:

- `load_from_uploaded` — keyed on uploaded `(filename, bytes)` tuples
- `build_cached_derived_bundle` — keyed on `(loaded_bundle, min_shift, max_shift, n_peaks, n_peaks_by_serotype_items)`
- `apply_cached_filters` — keyed on `(derived_bundle, serialized_filter_state)`
- `build_cached_sensor_assessment_artifacts` — keyed on `(filtered_features, selection)`
- `build_cached_single_sensor_consistency_artifacts` — keyed on `(filtered_features, selection)`
- `build_cached_global_qa_artifacts` — keyed on `(filtered_features, feature_columns)`
- `build_cached_classification_dataset` — keyed on `(filtered_features, excluded_map_policy, inlier_feature)`
- `build_cached_classification_artifacts` — keyed on `(clean_classification_df, feature_columns)`
- `build_cached_concentration_regression_dataset` — keyed on `(filtered_features, excluded_map_policy, inlier_feature)`
- `build_cached_global_regression_artifacts` — keyed on `(regression_clean, feature_columns)`
- `build_cached_two_stage_regression_artifacts` — keyed on `(regression_clean, feature_columns)`
- `build_cached_mtl_regression_artifacts` — keyed on `(regression_clean, feature_columns)`

Small helper functions and plot rendering are still done on rerun. The goal here is to cache the expensive dataframe and model-preparation steps, not every line of the UI.

"Reload Data" clears all caches and session state via `clear_app_data`.

### 14.4 Contract DTOs (`application/contracts.py`)

The application layer uses typed dataclasses as data transfer objects. Some are frozen where immutability is useful; others are mutable plain dataclasses because they hold evolving dataframe bundles.

| DTO | Purpose |
|-----|---------|
| `LoadedDataBundle` | Raw wide + tidy DataFrames from upload |
| `PeakArtifacts` | Peak infos by serotype, mean spectra, default serotype, raman_x |
| `DerivedDataBundle` | Trimmed/preprocessed DataFrames + peak artifacts |
| `FilterSelection` | Per-column selected values + exclude flag |
| `FilterCatalog` | Column ordering for main vs. "more" filters |
| `FilteredBundle` | Filtered tidy + features DataFrames + spectrum count |
| `SensorAssessmentSelection` | User's serotype, concentration, feature, method choices |
| `SensorAssessmentArtifacts` | Consistency, degradation, batch results + display tables |
| `ModelConsistencySelection` | Sensor, serotype, feature for single-sensor regression |
| `SingleSensorConsistencyArtifacts` | Regression result + zero-CFU baseline + model DataFrame |
| `GlobalQaArtifacts` | QA table + excluded sensor map |
| `OverlayArtifact` | Per-serotype/feature overlay data for multi-sensor plot |
| `ClassificationArtifacts` | Best/RF/SVM classification results |
| `GlobalRegressionArtifacts`, `TwoStageRegressionArtifacts`, `MtlRegressionArtifacts` | Concentration regression runs + figures for PDF export |

---

## 15. Package map: `sensd_sers_analysis`

```text
src/sensd_sers_analysis/
├── __init__.py                           # Curated public re-exports
├── modeling.py                          # Strict features, fold-local Pipelines and selection
├── splits.py                            # Shared sensor holdout and group validation
├── data/
│   ├── __init__.py
│   ├── io.py                             # Excel parsing, wide/tidy conversion
│   └── txt_converter.py                  # TXT, metadata/preset and workbook serialization
├── processing/
│   ├── __init__.py
│   ├── alignment.py                      # Master-grid snap + Raman shift trimming
│   ├── features.py                       # max/mean/integral extraction
│   ├── pca_features.py                   # Exploratory PCA on complete spectra
│   ├── peak_features.py                  # Serotype-specific peak detection
│   ├── targeted_peak_features.py        # Fixed-anchor peak heights for UI workflow
│   ├── metadata.py                       # log_concentration, concentration_group
│   └── filters.py                        # Cascading metadata filter logic
├── assessment/
│   ├── __init__.py
│   ├── outliers.py                       # IQR and z-score detection
│   ├── consistency.py                    # CV and replicate consistency
│   ├── degradation.py                    # Linear trend / temporal analysis
│   ├── sensor_assessment_regression.py   # Two-pass regression, global QA
│   └── batch_variance.py                 # Inter-sensor z-score analysis
├── classification/
│   ├── __init__.py
│   ├── data_prep.py                      # Identity eligibility and diagnostic response screening
│   ├── models.py                         # Random Forest + SVM training
│   └── plots.py                          # PCA scatter, confusion matrix, importance
├── regression/
│   ├── __init__.py
│   ├── data_prep.py                      # Regression-tab dataset assembly
│   ├── splits.py                         # Compatibility exports from shared splits.py
│   ├── metrics.py                        # Regression metric helpers
│   ├── models_global.py                  # Global (single-task) regressors
│   ├── models_two_stage.py               # Two-stage regression pipeline
│   ├── models_mtl.py                     # Multi-task / MtlSpectralNet
│   └── plots.py                          # Actual vs predicted, residuals, importances
├── config/
│   ├── __init__.py
│   ├── model_policies.py                 # QA / classification / training policy constants
│   ├── spectral_policies.py             # Spectral preprocessing policy hooks
│   └── targeted_peaks.py                # Default targeted-anchor positions (cm⁻¹)
├── application/
│   ├── __init__.py
│   ├── contracts.py                      # Typed DTOs for app layer
│   ├── dataset_pipeline.py               # Upload → derived bundle orchestration
│   ├── filtering_service.py              # Filter state serialization/application
│   ├── assessment_service.py             # Sensor assessment orchestration
│   ├── sensor_assessment_service.py      # Regression QA orchestration
│   ├── classification_service.py         # Classification dataset + training orchestration
│   ├── regression_service.py             # Regression paradigms + regression PDFs
│   ├── peak_discovery_service.py         # Peak discovery data preparation
│   └── targeted_peak_service.py          # Merge targeted peaks into filtered bundle
├── visualization/
│   ├── __init__.py
│   ├── plots.py                          # Spectra line plots (seaborn)
│   ├── stats.py                          # Feature distribution box/violin
│   ├── assessment_plots.py               # Degradation, batch, regression plots
│   ├── peak_discovery.py                # Peak anchor and signal-level plots
│   └── targeted_peak_plots.py           # Targeted peak verification figures
├── report/
│   ├── __init__.py
│   └── pdf_builder.py                    # ReportLab PDF assembly
└── utils/
    ├── __init__.py
    ├── labels.py                          # Column label formatting
    ├── natural_sort.py                    # Digit-aware sort keys
    └── parsing.py                         # Raman shift bound parsing
```

### `regression/` at a glance

This package holds **alternative concentration-regression experiments** (predict continuous dose proxies from spectra or engineered features), distinct from **classification** (`classification/`). Training, metrics, and plotting are organized per paradigm (`models_global.py`, `models_two_stage.py`, `models_mtl.py`). The Streamlit tabs **Regression V1–V3** call into `application/regression_service.py`, which orchestrates dataset construction and PDF generation without importing Streamlit inside `regression/`.

---

## 16. Module reference: `data/`

### `io.py`

- **`_parse_embedded_format(file_path)`** — Reads an Excel file in the embedded-metadata format. Locates concentration rows, prefers **Actual Concentration** for numeric alignment, parses file-level metadata and optional per-signal **File Name** / **Special Treatment** rows. Returns `(metadata_dict, raman_shift, signals_matrix, concentrations, per_signal_labels)`.
- **`_parse_file_metadata_block(...)`** / **`_parse_per_signal_label_rows(...)`** — Helpers for metadata vs. per-signal row extraction (supports legacy row order).
- **`_load_signal_file(file_path)`** — Wraps `_parse_embedded_format`; assembles metadata DataFrame + transposed signal DataFrame → single wide DataFrame. Raman columns named `rs_{value:.2f}`; adds `source_txt_filename` and `special_treatment` per signal.
- **`_collect_files(paths, pattern)`** — Resolves mixed file/folder paths to a flat list of `.xlsx` files.
- **`load_sers_data(paths, serotypes, pattern)`** — Public entry point. Loads, filters, concatenates.
- **`get_signals_matrix(df)`** — Extracts `(n_samples, n_wavenumbers)` NumPy array from wide DataFrame.
- **`get_raman_shift(df)`** — Extracts the wavenumber array from `rs_*` column names.
- **`get_metadata_columns(df)`** — Returns metadata-only DataFrame.
- **`wide_to_tidy(df)`** — Melts wide → long format with `raman_shift` and `intensity` columns.
- **`load_sers_data_as_wide_and_tidy(paths)`** — Returns `(wide_df, tidy_df, load_report)` with per-file outcomes.
- **`count_unique_spectra(df)`** — Counts unique `(filename, signal_index)` pairs.

**Constants:** `REQUIRED_METADATA_KEYS`, `META_COLS`, `RS_COL_PREFIX = "rs_"`, `RAMAN_SHIFT_DECIMALS = 2`.

---

## 17. Module reference: `processing/`

### `alignment.py`

- **`snap_spectra_to_master_grid(wide_df, dedupe_rtol, dedupe_atol)`** — Align all rows onto a session master Raman grid (largest-span spectrum wins). Overlap-localized linear interpolation; non-overlapping master shifts remain NaN.
- **`trim_raman_shift(wide_df, min_shift, max_shift)`** — Drops `rs_*` columns outside `[min_shift, max_shift]`. Preserves all metadata columns.

### `features.py`

- **`extract_basic_features(df_wide)`** — Computes `max_intensity`, `mean_intensity`, `integral_area`, `PC1`, `PC2` per sample. Joins PCA features from `add_pca_features`.
  Integral area uses the actual Raman coordinates, with units of intensity times Raman-shift units. Nonfinite integrals and spectra with fewer than two coordinates have `NaN` area and a logged warning; unavailable rows do not alter valid rows, and no sum or implicit imputation substitutes for the integral. PCA scores are fitted to the supplied dataset for exploration, not within classification training folds.
- **`get_available_feature_columns(df, peak_infos_by_serotype)`** — Returns ordered list of available feature columns (basic + fixed targeted peaks) in preferred display order.
- **`order_features_by_preference(features)`** — Sorts by `PREFERRED_FEATURE_ORDER`; extras appended alphabetically.
- **Constants:** `BASIC_FEATURE_COLUMNS`, `PREFERRED_FEATURE_ORDER`, `DEFAULT_GLOBAL_QA_FEATURES`, `CLASSIFICATION_FEATURE_BASE`.

### `pca_features.py`

- **`add_pca_features(df_wide, n_components=2)`** — StandardScaler + PCA on spectral matrix. Returns DataFrame with `PC1`, `PC2`, `PC1_var_ratio`, `PC2_var_ratio`.

### `peak_features.py`

- **`PeakWindowInfo`** — Dataclass: `peak_name`, `center`, `window_min`, `window_max`, `success_rate`.
- **`extract_dynamic_peak_features(df_wide, n_peaks, n_peaks_by_serotype, ...)`** — Full serotype-specific peak extraction pipeline. Returns `(features_df, peak_infos_by_serotype, mean_spec_by_serotype, default_serotype, raman_x)`.
- **`get_peak_height_columns(peak_infos)`** — Returns `["Peak_1_Height", ..., "Peak_K_Height"]` from a list of `PeakWindowInfo`.
- **Constants:** `ZERO_CFU_LABEL`, `OUTER_BASELINE_FRAC = 0.05`, `ANCHOR_PROMINENCE_FRAC = 0.01`, `ANCHOR_DISTANCE_INDICES = 15`.

### `targeted_peak_features.py`

- Computes heights at **fixed Raman-shift anchors** chosen in the Peak Feature Extraction tab (defaults from `config/targeted_peaks.py`), complementary to serotype-learned windows in `peak_features.py`.

### `metadata.py`

- **`preprocess_metadata(df)`** — Adds `log_concentration`, `concentration_group`, normalizes `date`.
- **`add_log_concentration(df)`** — `log10(conc)` for `conc > 0`; NaN for `conc ≤ 0`.
- **`add_concentration_group(df)`** — Nearest-bin assignment on log-scale.
- **`extract_scalar_concentration(series, df)`** — Handles list-per-row concentrations using `signal_index`.

### `filters.py`

- **`get_filter_options(df, ...)`** — Computes available filter values per column, respecting cascading constraints.
- **`filter_sers_data(df, selections)`** / **`filter_by_selections(df, selections)`** — Apply metadata filters (include/exclude mode).
- **`get_filterable_columns(df)`** / **`get_plot_hue_columns(df)`** / **`get_feature_metadata_columns(df)`** / **`pick_preferred_column(cols)`** — UI helpers for column selection in Streamlit widgets.
- **Constants:** `DEFAULT_FILTER_ORDER` includes `source_txt_filename` and `special_treatment` when present in loaded data (prep-generated workbooks).

---

## 18. Module reference: `assessment/`

### `outliers.py`

- **`detect_outliers_iqr(values, whis=1.5)`** — IQR fences. Returns boolean mask.
- **`detect_outliers_zscore(values, threshold=3.0)`** — Z-score cutoff. Returns boolean mask.
- **`filter_outliers(df, feature_col, method)`** — Dispatches and returns `(inliers_df, outliers_df)`.

### `consistency.py`

- **`ConsistencyResult`** — Dataclass: raw/filtered mean, std, CV, counts.
- **`coefficient_of_variation(values)`** — `σ / |μ|` as fraction.
- **`compute_consistency_metrics(df, feature_col, group_cols, outlier_method)`** — Per-group raw + filtered CV table.
- **`get_consistency_summary_table(df, feature_cols, group_cols)`** — Multi-feature summary.

### `degradation.py`

- **`DegradationResult`** — Dataclass: slope, intercept, R², p-value, stderr, interpretation.
- **`prepare_degradation_data(df, feature_col)`** — Aggregates to one value per `(sensor_id, test_id)` with `test_ordinal`.
- **`compute_degradation(df, feature_col, sequence_col, group_cols)`** — `scipy.stats.linregress` per group.
- **`add_sequence_column(df)`** — Ensures a numeric sequence exists for the X-axis.

### `sensor_assessment_regression.py`

- **`ConcentrationRegressionResult`** — Single-pass: slope, intercept, R², RMSE, fitted values.
- **`CleanedRegressionResult`** — Two-pass: raw + clean metrics, outlier mask.
- **`fit_concentration_regression(df, feature_col)`** — Single OLS fit on `log_concentration` vs. feature.
- **`fit_concentration_regression_cleaned(df, feature_col)`** — Two-pass with residual IQR outlier removal.
- **`get_zero_cfu_baseline(df, feature_col)`** — Mean feature for explicit rinsate controls.
- **`get_global_model_consistency(df, feature_cols)`** — Raw metrics for all `(sensor, serotype, feature)` combinations.
- **`get_global_model_consistency_qa(df, feature_cols)`** — Full QA pipeline with dual-threshold exclusion.
- **`MacroRegressionResult`** — Pooled batch-level: two-pass with macro outlier detection.
- **`compute_macro_batch_regression(df, serotype, feature_col, pass_sensors)`** — Pools Pass-sensor inliers and fits a single batch regression.

### `batch_variance.py`

- **`compute_batch_variance(df, feature_col, sensor_col, group_cols)`** — Per-sensor stats + z-score from batch.
- **`identify_deviating_sensors(batch_df, z_threshold=2.0)`** — Flags sensors with `|z| > threshold`.

---

## 19. Module reference: `classification/`

### `data_prep.py`

- **`label_classification_dataset(df)`** — Identity-eligible model rows without response screening.
- **`prepare_classification_dataset(df, excluded_map, feature_cols, inlier_feature)`** — Retrospective exclusion + pooled residual diagnostics with dynamic class labeling.

### `models.py`

- **`ClassificationResult`** — Pipeline, held-out predictions/metrics, training CV score, outer positions/seed and feature importances; external scaler is None.
- **`train_classifiers(df, feature_cols, target_col)`** — 80/20 sensor holdout → fold-local Pipeline scaling/tuning → RF + SVM → two `ClassificationResult` objects.

### `plots.py`

- **`plot_pca_classification(df)`** — 2D scatter of PC1 vs. PC2 colored by target.
- **`plot_confusion_matrix(result)`** — Heatmap of confusion matrix.
- **`plot_feature_importance(result)`** — Horizontal bar chart of RF feature importances.

---

## 20. Module reference: `config/`

### `model_policies.py`

Centralized policy constants. No logic—only values:

| Constant | Value | Used by |
|----------|-------|---------|
| `GLOBAL_QA_REJECTION_MULTIPLIER` | 2.0 | `sensor_assessment_regression.py` — RMSE exclusion threshold |
| `GLOBAL_QA_R2_MIN_THRESHOLD` | 0.80 | `sensor_assessment_regression.py` — R² exclusion threshold |
| `GLOBAL_QA_IQR_WHIS` | 1.5 | `sensor_assessment_regression.py` — Residual outlier IQR multiplier |
| `BATCH_DEVIATION_Z_THRESHOLD` | 2.0 | `application/assessment_service.py` — Deviating sensor z-score cutoff used in app-facing batch summaries |
| `CLASSIFICATION_INLIER_FEATURE` | `"integral_area"` | `classification_service.py` — Feature for inlier filtering |
| `CLASSIFICATION_QA_FEATURES` | `("integral_area",)` | `classification_service.py` — Features for exclusion map |
| `CLASSIFICATION_TEST_SIZE` | 0.2 | `models.py` — Train/test split ratio |
| `CLASSIFICATION_RANDOM_STATE` | 42 | `models.py` — Reproducibility seed |
| `CLASSIFICATION_RF_N_ESTIMATORS` | 100 | `models.py` — Random Forest tree count |

### `spectral_policies.py` / `targeted_peaks.py`

Supporting constants for spectral workflows. `spectral_policies.py` exports master-grid dedupe tolerances (`SNAP_SHIFT_DEDUPE_RTOL`, `SNAP_SHIFT_DEDUPE_ATOL`). `targeted_peaks.py` holds default targeted-anchor positions in cm⁻¹. Import these modules rather than duplicating literals in tabs or processing code.

---

## 21. Module reference: `application/`

The application layer bridges `src/` domain modules and the `apps/` Streamlit frontend. In practice it is mostly orchestration, DTO assembly, and serialization. The domain calculations still live in `processing/`, `assessment/`, `classification/`, and `regression/`.

### `contracts.py`

Typed dataclasses for all inter-layer data transfer (see §14.4 for the complete list).

### `dataset_pipeline.py`

- **`load_uploaded_bundle(files_data)`** — Validates simple, unique filenames before writing bytes to a temporary directory, then parses into `LoadedDataBundle`. Case/Unicode-equivalent names are rejected because spectrum identity is `(filename, signal_index)`.
- **`build_derived_bundle(loaded_bundle, min_shift, max_shift, n_peaks, ...)`** — Full preprocessing: `preprocess_metadata` → **`snap_spectra_to_master_grid`** → `trim_raman_shift` → `extract_basic_features` → `extract_dynamic_peak_features` → `DerivedDataBundle`.

### `filtering_service.py`

- **`build_filter_catalog(tidy_df, main_filter_count)`** — Identifies filterable columns and splits into main/more.
- **`compute_filter_options(tidy_df, filter_columns, current_state)`** — Cascading filter options.
- **`serialize_filter_state(state)` / `deserialize_filter_state(data)`** — For cache key stability.
- **`apply_filters(derived_bundle, filter_state)`** — Applies filters → `FilteredBundle`.

### `assessment_service.py`

- **`build_sensor_assessment_artifacts(filtered_features, selection)`** — Orchestrates consistency, degradation, batch analysis → `SensorAssessmentArtifacts`.
- **`build_sensor_assessment_pdf_bytes(artifacts)`** — Generates PDF via ReportLab.

### `sensor_assessment_service.py`

- **`build_single_sensor_consistency_artifacts(filtered_features, selection)`** — Two-pass regression for one sensor × serotype → `SingleSensorConsistencyArtifacts`.
- **`build_global_qa_artifacts(filtered_features, feature_columns)`** — Full QA table + exclusion map → `GlobalQaArtifacts`.
- **`build_overlay_artifacts(filtered_features, serotypes, features, excluded_map)`** — Multi-sensor regression overlay data.
- **`build_sensor_assessment_qa_pdf_bytes(...)`** — Sensor assessment QA report PDF.

### `classification_service.py`

- **`build_classification_clean_dataset(filtered_features, excluded_map_policy, inlier_feature)`** — Identity labeling without response-dependent exclusions; legacy function name retained for callers.
- **`run_classification_training(clean_classification_df, feature_columns)`** — `train_classifiers` → selects by training CV weighted F1 (RF reference when CV is unavailable) → `ClassificationArtifacts`.
- **`build_classification_report_pdf_bytes(classification_artifacts)`** — Serotype classification report PDF.

### `regression_service.py`

- **`build_concentration_regression_dataset(...)`** — Identity-eligible bacterial rows with finite positive measured actual CFU; no retrospective screening.
- **`run_global_concentration_regression(...)`** / **`run_two_stage_concentration_regression(...)`** / **`run_mtl_concentration_regression(...)`** — Train/eval paradigms from `regression/` and bundle artifact DTOs.
- **`build_global_regression_pdf_bytes(...)`** / **`build_two_stage_regression_pdf_bytes(...)`** / **`build_mtl_regression_pdf_bytes(...)`** — Regression PDF assembly via ReportLab.

### `targeted_peak_service.py`

- **`merge_targeted_peaks_into_filtered_bundle(...)`** — Adds targeted peak-height columns to the filtered bundle used by downstream tabs.

### `peak_discovery_service.py`

- **`build_peak_diagnostic_context(...)`** — Aligns wide/features DataFrames for signal-level peak verification.
- **`build_peak_anchor_overviews(peak_artifacts)`** — Per-serotype mean spectrum + peak info summaries.
- **`build_signal_selection_options(...)` / `build_matching_signal_options(...)` / `build_signal_verification_artifact(...)`** — Streamlit widget support for interactive peak inspection.

---

## 22. Module reference: `visualization/`

The visualization modules are mostly matplotlib/seaborn wrappers intended to be safe to import from notebooks or scripts. One exception is `assessment_plots.py`, which intentionally imports regression-related helpers from `assessment.sensor_assessment_regression` so it can render those views directly.

### `plots.py`

- **`plot_spectra(df, hue, style, show_variance, errorbar, figsize)`** — Seaborn lineplot of tidy-format spectra. Supports numeric hue with continuous colorbar, categorical hue with legend, and variance display (SD, SE, CI).
- **`VARIANCE_OPTIONS`** — List of `(label, show_variance, errorbar)` tuples for UI radio buttons.

### `stats.py`

- **`plot_feature_distribution(df, feature_col, x, hue, plot_type, figsize)`** — Box or violin plot with overlaid strip plot.

### `assessment_plots.py`

- **`plot_degradation_trend(df, feature_col, sequence_col, group_col)`** — Scatter + regression line per sensor.
- **`plot_batch_boxplot(df, feature_col, sensor_col, group_col)`** — Boxplot per sensor, with optional hue stratification.
- **`plot_concentration_regression(df, feature_col, regression_result, ...)`** — Scatter + raw/clean regression lines + zero-CFU baseline + outlier markers.
- **`plot_multi_sensor_regression(df, serotype, feature_col, excluded_sensors)`** — Multiple regression lines (solid = pass, dashed gray = excluded).
- **`plot_macro_batch_regression(df, serotype, feature_col, pass_sensors)`** — Pooled regression with macro outlier markers.

### `peak_discovery.py`

- **`plot_peak_anchor_summary(raman_x, mean_spectrum, peak_infos, serotype, ...)`** — Mean spectrum with shaded peak windows and anchor markers.
- **`plot_signal_level_peak_verification(verification_artifact, ...)`** — Single-spectrum plot with per-peak windows and detected peak markers.

### `targeted_peak_plots.py`

- Figures for verifying heights at user-selected anchor positions (Peak Feature Extraction workflow).

---

## 23. Module reference: `report/`

### `pdf_builder.py`

ReportLab-based PDF generation. Public builders take precomputed tables and figures rather than recomputing analytics:

- **`build_sensor_assessment_pdf(...)`**
- **`build_sensor_assessment_qa_pdf(...)`**
- **`build_classification_report_pdf(...)`**
- **`build_regression_concentration_pdf(...)`** — Consolidated regression-tab report (metrics + diagnostic figures).

Internal helpers: `_df_to_table_data`, `_compute_table_col_widths`, `_figure_to_image_bytes`.

---

## 24. Module reference: `utils/`

### `labels.py`

- **`format_column_label(col)`** — Converts `snake_case` column names to title case and preserves a small built-in acronym set (`id`, `cfu`, `uuid`, `url`).

### `natural_sort.py`

- **`natural_sort_key(s)`** — Key function for digit-aware sorting (e.g., `"10 CFU" > "1 CFU"`).
- **`natural_sort(items)`** — Sorted list using natural sort.
- **`order_concentration_labels(labels)`** — Applies natural sort to concentration group labels.

### `parsing.py`

- **`parse_raman_shift_bound(s)`** — Parses a string to `float | None`; empty or invalid → `None`.

---

## 25. Streamlit app: `apps/`

### `app.py`

Entry point. Mode switch on `_app_mode`: prep delegates to `txt_to_excel.render_prep_mode()`; analysis renders sidebar upload, Raman shift, filters, and **ten** tabs. When upload is empty, shows `st.title("SERS Data Explorer")` plus a **Next steps** checklist (upload workbook, optional TXT prep path, trim/filter, tab workflow). After data loads, the same title precedes the filtered-sample caption and tab strip.

### `txt_to_excel.py`

Prep-mode widgets and session-state ownership live in `apps/txt_to_excel.py`. Independently callable TXT parsing, finite metadata validation, JSON setup serialization and embedded-workbook generation live in `data/txt_converter.py`; shared field order/keys live in `config/metadata_schema.py`. Invalid measurement rows fail rather than disappear, grids compare with absolute tolerance and no relative slack, and duplicate upload names fail before merge. Date/time inputs preserve recorded meaning rather than truncate malformed compact values. Output bytes/preview belong to the exact uploaded bytes, ordered per-signal values, bounds and output name; changed inputs invalidate downloads. Reload/shared-field persistence and explicit filename-keyed categorical presets retain their distinct scopes.

### `cache.py`

`@st.cache_data` wrappers for the main repeated app-level computations (see §14.3).

### `state.py`

Session-state management: widget key generation, filter state clearing, peak artifact persistence, full UI reset.

### `theme.py`

UI constants: figure sizes, slider limits, matplotlib aesthetics, HTML dividers.

### `components/`

- `data_loading.py` — Upload handler with `clear_app_data`.
- `filter_ui.py` — Dynamic filter rendering with pills/multiselect, exclude toggle, per-filter and global reset.
- `raman_sidebar.py` — Raman shift bounds and per-serotype peak count controls.
- `shared_ui.py` — PDF generate/download section, metric row, stretch-width DataFrame/figure rendering.

### `tabs/`

Each tab module exposes a `render(...)` function. Most tabs receive the filtered feature dataframe and, where needed, peak artifacts or the wide dataframe:

- `spectra_viewer.py` — Hue, style, variance controls → `plot_spectra`.
- `peak_discovery.py` — Anchor overview + signal-level peak verification.
- `peak_feature_extraction.py` — Targeted anchors (session state) + dynamic peak tooling.
- `feature_analysis.py` — Feature distributions, correlation heatmap, vs log concentration.
- `sensor_qc_legacy.py` — Legacy CV-style consistency, degradation, batch stability + PDF.
- `sensor_assessment.py` — Single-sensor regression, global QA table, multi-sensor overlay, macro batch, sensor-assessment QA PDF.
- `serotype_classification.py` — PCA scatter, RF/SVM training, confusion matrix, feature importance, classification report PDF.
- `regression_global.py`, `regression_two_stage.py`, `regression_mtl.py` — Concentration regression paradigms + PDF callbacks (via `regression_service`).
- `regression_common.py` — Shared widgets/helpers for regression tabs.

---

## 26. Tests

Under `tests/`:

- **`test_application_services.py`** — Integration tests for the application layer: synthetic `LoadedDataBundle` construction, `build_derived_bundle` (including master-grid snap parity), `apply_filters`, `build_sensor_assessment_artifacts`, sensor assessment regression / classification parity.
- **`test_embedded_io_per_signal_rows.py`** — Round-trip tests for File Name / Special Treatment rows and Actual-vs-Target concentration parsing via minimal prep-built workbooks.
- **`test_txt_to_excel_template.py`** — Prep metadata preset JSON import/export, validation helpers, reload persistence behavior (`RELOAD_PERSIST_WIDGET_KEYS` / `RELOAD_CLEAR_WIDGET_KEYS`), merged-preview legend formatting, and date/time coercion helpers.

Current test coverage is still light relative to the full surface area, but prep/I/O edges now have dedicated modules beyond the application-service parity test.

The `tests/test_application_services.py` module is compatible with both **`pytest`** and **`python -m unittest`**. Floating-point assertions use pandas/numpy-aware comparisons (`assert_frame_equal`, `np.allclose`) where appropriate.

---

## 27. Settings ownership and reproducibility

`config/model_policies.py` owns shared QA thresholds, model/search settings, split seeds, and Validation evaluation guards. `config/preprocessing.py` owns the reference exposure and App normalization default. Spectral grid tolerances and default fixed anchors have their own shared config modules.

Function parameters and local constants own settings used by one calculation, such as the degradation stability threshold and peak-detection defaults. A value belongs in shared configuration when multiple callers need the same policy; centralized storage is not a requirement for every numeric literal.

MTL's `random_state` controls the training-sensor inner split and CPU PyTorch initialization, dropout, and minibatch shuffling. Training restores the caller's CPU RNG state. Repeatability applies within the same runtime/hardware; it is not a cross-platform numerical guarantee. The inner seed is outer seed +17; inner source positions and fine-tuning budget are returned with the output, alongside outer split artifacts.

---

## 28. Design choices, limitations & writing-up guidance

### 28.1 Design choices

1. **Wide-format as canonical representation.** The wide format (one row per spectrum, one column per wavenumber) enables fast NumPy vectorized operations. Tidy format is derived for plotting only. This means column names encode data (Raman shift values), which is unusual but enables the `rs_` prefix convention for programmatic column selection.

2. **Serotype-specific peak detection.** Exploratory anchors are learned separately per serotype from explicit bacterial samples. Controls do not supply anchors. The efficacy of this representation for held-out prediction requires training-only evaluation.

3. **Two-pass regression for robustness.** The residual IQR approach for outlier detection in regression (§11.3) is more targeted than applying IQR to the raw feature values. A point can be an outlier *relative to the regression model* (high residual) without being extreme in the feature distribution. This distinction matters when the feature naturally varies with concentration.

4. **Dual-threshold sensor exclusion.** Condition A (RMSE-based) catches noisy sensors. Condition B (R²-based) catches dead/flat sensors. Neither alone is sufficient: a flat sensor has low RMSE (always predicts the mean) but also low R² (no dose-response relationship).

5. **Application layer separation.** The `application/` package exists to prevent Streamlit-specific concerns (caching, session state, widget data shapes) from leaking into domain logic. Domain modules (`assessment`, `classification`, etc.) have no knowledge of Streamlit. The TXT prep UI owns widgets/state while reusable parsing and export code lives in `data/txt_converter.py`. Standalone figures avoid pyplot registry leaks on plot failures; report requests release their own figures in success/failure paths. PDF tables retain full headers and values, wrap safe Unicode text and repeat headers; figures preserve aspect ratio. Expected missing diagnostic sections show a reason, while unexpected errors abort generation. Downloads are bound to the current dataframe and settings.

6. **Master grid before trim.** Snapping heterogeneous Raman grids onto a session master axis (§7.1) runs before sidebar trimming so PCA and peak features operate on aligned columns. NaN outside per-sensor overlap avoids silent extrapolation.

### 28.2 Limitations

1. **Input preprocessing provenance.** The pipeline does not implement additional spectral baseline subtraction. LJ identifies current New Format inputs as provider-corrected Clean Peaks; see `DATA_SCOPE.md`. Integral area includes any remaining baseline contribution. Do not infer raw instrument provenance from the Raman Shift coordinate label.

2. **No cosmic ray removal.** Single-point intensity spikes from cosmic rays are not explicitly handled. They may inflate `max_intensity` and skew peak detection. The outlier detection at the feature level (§9) provides indirect mitigation.

3. **Linear dose-response assumption.** The regression model assumes `feature = a × log₁₀(concentration) + b`. If the true dose-response is nonlinear (e.g., saturation at high concentrations), the linear model will underperform. R² values should be interpreted with this limitation in mind.

4. **Retrospective evaluation and independence.** Classification, regression and Validation hold out recorded sensor IDs, with training-only scaling/selection and no QA-based model exclusions. Validation repeats five rounds. Recorded IDs do not establish physical-device or biological-preparation independence. Historical inspection and manual feature choices remain retrospective; neither these scores nor QA establish prospective qualification. Separate task/service splits are not automatically paired cross-task comparisons.

5. **No spectral deconvolution.** Overlapping peaks are not decomposed. The peak height extraction uses a simple max-within-window approach (§8.5), which may underestimate heights for partially overlapping peaks.

### 28.3 Writing-up guidance

For journal papers based on this codebase:

- **Methods section:** Describe the pipeline stages (§6–§13) in the order presented. Reference specific functions/modules for reproducibility. The two-pass regression (§11.3) and dual-threshold exclusion (§12.1) are the project-specific parts worth explaining carefully.
- **Figures:** The Streamlit app can generate most of the exploratory and reporting figures you would likely reuse in a paper: spectra overlays, peak diagnostics, degradation trends, regression plots with outliers marked, confusion matrices, and feature importance charts.
- **Reproducibility:** The main shared random seeds, thresholds, and hyperparameters are documented in §27, but that section should be read as a practical map rather than a guarantee that every numeric constant is already centralized.
- **Limitations paragraph:** §28.2 provides an honest basis. Acknowledge the linear model assumption and single-split evaluation.

---

## 29. Known gaps & optional extensions

**Already in the tree:** Full pipeline from Excel ingestion to classification, instrument **TXT → embedded Excel prep mode**, serotype-specific peak detection, master-grid Raman alignment, two-pass regression QA, dual-threshold sensor exclusion, macro batch regression, concentration-regression paradigms (V1–V3), PDF reports, and a working Streamlit frontend for exploration and reporting.

**Open, depending on project goals:**

- **Baseline correction** (SNIP, polynomial, or ALS) as an optional preprocessing step before feature extraction.
- **Repeated outer sensor holdout** for classification evaluation to replace or supplement the single 80/20 split.
- **Additional classifiers** (e.g., gradient boosting, logistic regression) for comparison.
- **Spectral deconvolution** for overlapping peak resolution.
- **Noise injection / robustness testing** for sensor simulation.
- **Multi-laboratory generalization** studies using data from different instrument configurations.
- **Automated peak count selection** (currently user-specified per serotype).
- **Vectorized peak extraction** — the current per-row loop in `extract_dynamic_peak_features` is correct but could be vectorized for performance on large datasets.
