# SENS-D Project Instructions

## 1. Working Approach

Use current source and task-relevant documentation to understand the work. Preserve intended goals and justified constraints; choose methods, tools, and structure according to current needs and evidence. Existing layouts and implementations are not permanent requirements.

Prefer clear, cohesive code. Reuse existing functionality where suitable; introduce abstractions, dependencies, or compatibility handling for concrete needs. Use validation where external inputs or independently callable APIs require it; internal helpers may rely on valid upstream guarantees. Avoid speculative fallbacks and repeated checks without a distinct purpose.

Give adjustable settings clear ownership, defaults, and override behavior. Choose function parameters, local constants, or shared configuration according to actual use.

## 2. Scientific Correctness

For data scope, provenance, or metadata semantics, consult the relevant sections of `docs/DATA_SCOPE.md`. For numerical methods, models, or evaluation, consult `docs/THEORY_AND_IMPLEMENTATION.md` and affected API contracts. Resolve disagreements between intended requirements and current behavior explicitly; neither code nor documentation alone establishes scientific correctness.

Preserve raw measurements and metadata. Make transformations, exclusions, units, missing-data handling, and assumptions explicit. Do not disguise invalid input or substitute a different mathematical quantity merely to complete execution.

Use explicit Sample Type for sample identity; do not infer control labels or initial target concentrations from actual concentration.

Preserve intended behavior during refactoring. Explain and validate intentional changes, especially those affecting scientific meaning. Match evaluation to the intended independent unit and prevent data leakage; learned preprocessing and model selection must respect training and held-out partitions.

Keep consequential results reproducible through relevant provenance, effective parameters, seeds, and split definitions. Choose precision and optimization from numerical requirements and demonstrated needs. Support non-obvious scientific choices with appropriate sources or validation.

## 3. Code and Documentation

Use `pyproject.toml` for dependencies and formatter/linter settings. Follow consistent naming, absolute package imports, and practical type hints.

Use NumPy conventions for structured docstrings. Simple helpers may use one-line docstrings. Explain non-obvious contracts, units, shapes, assumptions, and side effects; avoid repeating obvious code or rewriting unrelated documentation solely for style.

Keep affected documentation and comments accurate. Scientific figures should clearly identify quantities, units, and series.

## 4. Verification and Changes

Retain reviewed tests for consequential behavior, mathematical expectations, and regressions. Prefer independent expected results and justified numerical tolerances; revise expectations explicitly for intentional behavior changes. Use small real files or mocks according to the contract being tested.

Run relevant configured checks, expanding validation when risks or failures justify it. Resolve underlying issues rather than weakening checks; necessary exceptions should be narrowly scoped and justified.

Update affected callers, tests, and documentation together. Use `git mv`/`git rm` for tracked moves/deletions. When committing, capitalize the type prefix and first word, for example `FIX: Preserve integral semantics`.
