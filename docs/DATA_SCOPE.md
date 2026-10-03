# Research data scope and provenance

## Current analysis inputs

Paths below are relative to the repository root. Use the existing embedded-metadata loader in `src/sensd_sers_analysis/data/io.py`.

| Provider collection | Directory | Current interpretation |
| --- | --- | --- |
| Adheesha, 2025 | `example_data/Adheesha - New Format/_Data for ML-New-2025/Dilutions` | Dilution experiments in the embedded-metadata format |
| Adheesha, 2026 | `example_data/Adheesha - New Format/_Data for ML New-2026/Dilutions` | Same experiment family, next year |
| Amjad | `example_data/Amjad - New Format` | LJ identifies this collection as dilution, baseline-corrected data without separate experiment subfolders |

The active directories mirror the three SharePoint collections verified on 2026-09-17. The verified duplicate source snapshots were deleted at LJ's request; the active workbooks and complete pre-cleanup example backup remain. At source reconciliation, all 125 workbook contents matched their pre-sync versions and two S16 filenames differed. Subsequent metadata migration is documented below. Detailed source comparisons and QA accounting are in `reports/data_inventory_2026-09-17/README.md`.

LJ identifies the current inputs as **Clean Peaks**, spectra already baseline-corrected by the providers, rather than raw instrument output. Preserve this provenance; do not silently apply another baseline correction. This processing history is supplied by LJ, not established merely by reading numeric intensities.

The format places metadata inside each workbook: sensor geometry/model, integration time, scan average, sensor/test/connection IDs, serotype, rinsate type, experimental date/time, operator and notes. Per-spectrum fields include source TXT filename, special treatment, target concentration and actual concentration. Representative workbooks from all three directories were inspected on 2026-09-17. New Format describes the schema, not the experimental date: reformatted collections include 2025 measurements. LJ recalls agreeing on the schema around spring 2026; the precise date is unconfirmed.

## Experiment and representation boundaries

Adheesha's New Format collection also contains `Heat kill`, `PAA`, and a 2025 `Repeatability` directory. The current dilution analysis does not include those folders. LJ identifies heat-kill and PAA as Salmonella inactivation experiments, separate from the current ST/SE question. Do not infer a specific chemical expansion for PAA without its protocol.

The older `data/SERS Data 9 (Jun 2026)/original/Adheesha- ML Excel Files` collection includes experiment-specific directories and, in several branches, both `Clean peaks` and `Raman shift`. LJ identifies these as baseline-corrected and raw instrument representations respectively. Do not combine both as independent observations. Legacy collections (`Amjad - ML`, the older Adheesha files, and `SERS Data 9 (July-December 2025) - test`) are not additional inputs to the current New Format analysis without an explicit reconciliation.

A workbook column labelled **Raman Shift** is the spectral coordinate axis. Its presence does not mean that a New Format workbook contains the raw representation from a directory named `Raman shift`.

## Scope and counting units

Across all categories/dates, the 125 active files contain 580 spectra and 53 recorded IDs through 2026-05-07. Since 2025-09-01, all categories contain 506 spectra/47 IDs; Salmonella ST/SE/SH/SM/MIX alone contains 486/45 through May 6. The 20 E.COLI spectra/2 IDs on May 7 are separate. Thus the February ST/SE endpoint does not indicate missing broader March-May data. Current filenames conflict with embedded serotypes in S16-SE-T2 (embedded ST) and four May 7 SM-named files (embedded E.COLI). Preserve embedded values pending provider confirmation.

For the current Almasri ST/SE question, distinguish all-date New Format dilution coverage from embedded experimental dates on or after **2025-09-01**. Date and experiment restrictions define eligibility before QA; they are not evidence of sensor failure. Other serotypes are present in these folders and belong to broader analyses only when explicitly selected.

A 2026-09-17 inventory using the existing core loader verified 36 unique sensor IDs, 76 workbooks and 323 spectra for ST/SE across all dates (2025-07-01 through 2026-02-12). With embedded date >= 2025-09-01: 30 IDs, 62 workbooks and 259 spectra, covering 2025-09-09 through 2026-02-12. Adheesha contributes 20 IDs/39 files/164 spectra; Amjad contributes 11 IDs/23 files/95 spectra; their shared ID is AM1. Across all dates, counts are Adheesha 25 IDs/50 files/214 spectra and Amjad 13 IDs/26 files/109 spectra, with AM1 and S142 shared. No QA or model filtering was applied. After the date cutoff, the 153 observed sensor/serotype/target-concentration combinations have spectrum counts distributed as: 63 with one, 82 with two, 4 with three, 1 with four, 2 with five and 1 with six. These are observed combinations, not a complete experimental-design grid. AM1's shared spelling does not establish that both collections refer to one physical device. Almasri's 42-device total remains unreconciled. The received counts include a verified exact duplicate of Amjad S190-SE-T2 (4 spectra); distinct-content ST/SE counts after cutoff are 255 spectra and 30 IDs. The separate de-duplicated audit changes QA weighting; consult the report rather than subtracting four from final model counts.

- **File:** an input workbook; filename alone may not be globally unique across folders.
- **Spectrum:** one intensity trace, represented as one row in the loader's wide dataframe. Raman coordinate rows in tidy data are not additional spectra.
- **Sensor ID:** a recorded identifier, not independently validated physical-device identity.
- **Test/connection:** identifiers interpreted with sensor, date and other context; T1 alone is not a globally unique experiment.
- **Replicates:** report observed spectra per sensor/serotype/target concentration; do not equate them with independent biological preparations.
- **Dates:** embedded experimental dates describe measurement coverage, not receipt dates or a verified delivery cutoff.
- **Operator/provider:** embedded operator is distinct from the person or collection that delivered the file.

Report loaded scope, active filters, missing metadata, load failures and cross-source identifier overlap alongside counts. Keep inventory before QA distinct from model eligibility, QA flags, removed spectra and final training/test datasets.

## Storage and backup boundaries

`example_data/` contains the workbook collections `Adheesha - New Format/` and `Amjad - New Format/`, plus manual converter TXT inputs under `txt_to_excel/`. Adheesha retains its original year and experiment subdirectories, including Heat kill, PAA and Repeatability. There are 161 workbooks in total; the current Dilutions selection uses 125 of them. Loading the whole example directory is therefore not equivalent to the current research scope. Directory cleanup preserved workbook contents; the later explicit metadata migration is documented below. A fresh core-loader inventory after cleanup confirms 768 spectra and 59 distinct recorded sensor IDs across all 161 workbooks, dated 2025-07-01 through 2026-05-14. Folder collections contain Dilutions 125 files/580 spectra, Heat kill 17/51, PAA 15/75 and Repeatability 4/62. These collection boundaries describe the supplied directories, not a new scientific metadata requirement.

`data/SERS Data 9 (Jun 2026)/` is the complete pre-cleanup copy of `example_data/`, preserving its relative tree, including `original/`. At creation, all 529 files passed byte-hash comparison: 509 data workbooks, three Excel temporary lock files and 17 Finder metadata files. This backup is excluded from Git by `/data/`. Legacy workbook removals from `example_data/` are staged in Git; no commit or publication is implied.

The source reconciliation compared 125 downloaded workbooks with 125 pre-sync versions. The redundant `data/source_snapshots/` directory was deleted at LJ's request after verifying equal worksheet values/formulas. `source_comparison.json` preserves the historical paths, hashes and comparison outcomes; it is a historical record, not a claim that the deleted files remain available. Repeating that exact pre-sync comparison requires recovering its original inputs.

The September 17 audit's historical CSV/JSON source paths retain `example_data/original/...`. Their current active equivalents omit `/original`; their archived equivalents replace `example_data/` with `data/SERS Data 9 (Jun 2026)/`. The runnable audit configuration uses current active paths.

## Per-spectrum metadata schema

The converter preserves the 16 shared metadata rows and emits File Name, Sample Type, Special Treatment, Initial Target Concentration (CFU/mL), Actual Concentration (CFU/mL), then the Raman Shift/intensity block. Outputs are plain values with no dropdown validations or extra parameter rows. `target_concentration` remains the internal dataframe key for compatibility; its meaning is the initial target before special treatment. Actual concentration describes the final measured sample condition.

Generator Sample Type choices are Rinsate control and Bacteria sample. Special Treatment choices are None, Heat treated, PAA 10 ppm, PAA 50 ppm and PAA 100 ppm. None exports as a blank cell; preprocessing presents it as None for filtering. Every uploaded spectrum has independent selectors next to its concentration inputs. Choices are defined in `config/metadata_schema.py`.

Shared-field reload persistence is unchanged. Per-signal choices and concentrations reset when the upload batch changes or Reload Data is used. An explicitly imported template stores categorical signal labels by source filename; only matching filenames receive these labels. Concentrations are not stored in that preset. Old metadata-only JSON templates remain supported.

Sample Type, not a zero concentration, identifies controls in classification, control-versus-bacteria validation and peak-window routing. Blank or unrecognized identities remain visible in inventory but are excluded from sample-class labeling, control comparisons and peak learning; the App shows a warning. Older Excel labels remain readable, but legacy files without Sample Type need metadata migration for these analyses. Missing initial targets in explicit sample-type datasets remain Unknown rather than inheriting the actual-concentration group.

Nominal concentration groups use the exact recorded initial target, never bins inferred from actual concentration. For example, initial target 100 with actual 78, 103, or 230 stays in the 100 CFU/mL group; initial target 1 with actual zero remains a bacteria sample when Sample Type says so. The internal compatibility column `concentration_group` aliases `target_concentration_group`; neither is a separate sidebar or Spectra factor. Actual concentration remains the measured numerical value. Control baselines use explicit Sample Type. Sidebar filters expose the original fields with Exclude and Reset. The five main dimensions stay expanded regardless of option count. All other dimensions, including Excel and source TXT filenames, use a searchable dropdown when their complete loaded options average at least 24 characters and total at least 320 characters; otherwise they use expanded pills. This text-density heuristic is configured in `config/metadata_schema.py` and uses the full loaded option set so cascading filters do not change widget type.

Uploaded filenames must be simple, distinct basenames; case/Unicode-equivalent names are rejected before writes because App spectrum identity uses `(filename, signal_index)`. The loader retains arbitrary normalized metadata in both wide and tidy views. Duplicate metadata keys, normalized-name collisions, and names reserved for spectral/per-signal/computed columns fail explicitly rather than overwrite values; reservations are defined in `data/io.py`. Coordinates and intensities must be finite, and Raman coordinates must yield distinct labels at the existing two-decimal loader precision. Within the parsed spectral block, a missing coordinate on a row containing intensity data fails the file; fully blank separator rows may be ignored. Per-file failures remain visible in the load report. These checks leave source workbooks unchanged.

The scientific observation is one spectrum plus its acquisition, sample and treatment metadata. Folder membership preserves provenance, not scientific identity. Untreated data from another directory are not inherently unusable.

## Metadata migration record

The batch utility is `src/sensd_sers_analysis/data/metadata_migration.py`; scope and policy are in `config/metadata_migration.py`. Run `python -m sensd_sers_analysis.data.metadata_migration` for a plan, or append `--apply` to migrate. Already migrated files are skipped without overwriting the audit record. Each original must match `data/SERS Data 9 (Jun 2026)/original/`; all staged exports are checked before source replacement. The archive is unchanged.

All 161 workbooks were migrated on September 17: 768 spectra, including 580 Dilutions spectra; recent ST/SE remains 259 spectra/30 IDs. All spectral cells, actual concentrations, source TXT names and original shared metadata were preserved. The migration normalizes legacy treatment labels, adds Sample Type, renames the target row and fills 45 PAA initial targets with 1000 CFU/mL, as explicitly confirmed by LJ. Heat initial targets are recovered only where the legacy treatment text explicitly records the dose; seven remain unknown after the September 18 identity review. A one-time legacy untreated mapping uses existing target/actual/source labels; runtime analysis does not repeat that inference.

Detailed before/after values, mapping basis and hashes are in `reports/data_inventory_2026-09-17/metadata_migration.json`. Review entries are in `metadata_migration_review.csv` in the same directory:

- January 15 B5-Mix-T1 Heat: columns B/C have opposing treatment labels and source filenames.
- March 19 DI7-SH-T1/T2 Heat: column D says Untreated in metadata but Heat killed in its source filename.
- LJ confirmed on September 18 that source TXT filenames govern these four identities: B5 column B and both DI7 column D spectra are Bacteria sample / Heat treated; B5 column C is Rinsate control / no treatment. Actual concentrations and spectra are unchanged. B5 column B initial target is unknown because its previous zero accompanied an incorrect Rinsate label; column C has control target zero. Together with six other Heat spectra, seven initial targets remain pending. Original migration entries are retained; `resolutions_2026_09_18` records the confirmed corrections and current hashes.

The 2026-09-17 source-comparison and QA artifacts describe the pre-migration schema. Their hashes and source-cell-equality claims are historical. The active workbooks intentionally have different metadata and hashes after migration. No experimental model accuracy was rerun during migration.

## Resumption

Before a data run, state the input collections, experiment exclusions, serotypes, date boundaries and counting unit so LJ can verify the intended scope. Reuse the current app/core pipeline rather than introducing a separate parser. Consult this document when a request refers to current data, New Format, Clean Peaks, provider counts or the 42-device question.
