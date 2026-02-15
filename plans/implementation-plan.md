# Implementation Plan: Runner Optimization, TIV Extraction, and JSON Sidecars

## Context

Three improvements to the parcellate package:
1. The runner has suboptimal parallelism — atlas processing is serialized even when n_jobs > 1, and cross-subject parallelism lacks progress feedback.
2. CAT12 produces TIV (Total Intracranial Volume) in XML reports (`cat_*.xml`) but the package doesn't extract it.
3. Parcellation outputs lack provenance metadata — no record of which atlas, mask, or source file produced each TSV.

---

## Feature 1: Optimize Runners

### 1A. Cross-atlas parallelism in `runner.py`

**File:** `src/parcellate/interfaces/runner.py`

**Problem:** The outer loop iterates atlases sequentially, submitting scalar maps per atlas and blocking before moving to the next. With n_jobs > 1, cross-atlas work can't overlap.

**Change:** Split into two phases:
1. **Prepare phase** (sequential): Validate spaces, create VolumetricParcellator, call `.fit()` for each atlas
2. **Execute phase** (parallel): Submit ALL `_parcellate_scalar_map` calls across all atlases into one ThreadPoolExecutor, collect results via `as_completed()`

### 1B. Progress feedback for cross-subject parallelism

**Files:** `src/parcellate/interfaces/cat12/cat12.py`, `src/parcellate/interfaces/qsirecon/qsirecon.py`

**Problem:** Both use `[future.result() for future in futures]` which blocks in submission order with no progress logging.

**Change:** In `run_parcellations()`, replace with `as_completed()` pattern (like `cli.py` already does) with `[N/total]` progress logging. Add `as_completed` to imports.

---

## Feature 2: TIV Extraction for CAT12

### 2A. Add XML discovery and TIV extraction to loader

**File:** `src/parcellate/interfaces/cat12/loader.py`

Add two public functions (adapted from `examples/tiv.py`):
- `discover_cat12_xml(root, subject, session) -> list[Path]` — globs `cat_*.xml` in the anat directory using existing `_build_search_path`
- `extract_tiv_from_xml(xml_path) -> float | None` — parses `<subjectmeasures><vol_TIV>` with fallback search, returns mL value

New imports: `xml.etree.ElementTree`, `typing.Optional`

### 2B. Add TIV orchestration to `cat12.py`

**File:** `src/parcellate/interfaces/cat12/cat12.py`

Add functions:
- `_build_tiv_output_path(context, destination) -> Path` — returns `{dest}/cat12/sub-{id}[/ses-{id}]/anat/{label}_tiv.tsv`
- `_extract_and_write_tiv(root, context, destination) -> Path | None` — discovers XMLs, extracts TIV from each, writes a TSV with columns `source_file` and `vol_TIV`

Call `_extract_and_write_tiv` from `_run_recon()` after parcellation, appending the TIV path to outputs.

### 2C. Add TIV as column in parcellation TSVs

**File:** `src/parcellate/interfaces/cat12/cat12.py`

In `_run_recon()`, before calling `run_parcellation_workflow`:
1. Extract TIV value (first valid from XML files)
2. After getting `ParcellationOutput` results, add a `vol_TIV` column to each `result.stats_table`

This gives both a standalone TIV file AND TIV embedded in each parcellation TSV.

### 2D. Integrate TIV into `cli.py`

**File:** `src/parcellate/interfaces/cat12/cli.py`

In `process_single_subject()`, import and call `_extract_and_write_tiv` from `cat12.py`. Also extract TIV value and add to parcellation results (same pattern as 2C).

---

## Feature 3: JSON Sidecar Files

### 3A. Add shared sidecar writer to utils

**File:** `src/parcellate/interfaces/utils.py`

Add `write_parcellation_sidecar(tsv_path, original_file, atlas_name, atlas_image, atlas_lut, mask, space, resampling_target, background_label) -> Path`:
- Writes `{tsv_stem}.json` alongside the TSV
- Content matches `examples/associated_json.json` structure plus: `space`, `resampling_target`, `background_label`, `software_version` (from `importlib.metadata`), `timestamp` (UTC ISO)
- New imports: `json`, `datetime`

### 3B. Integrate into CAT12 `cat12.py`

**File:** `src/parcellate/interfaces/cat12/cat12.py`

Change `_write_output` signature to accept `config: Cat12Config`. After writing TSV, call `write_parcellation_sidecar(...)`. Update the call site in `_run_recon` to pass config.

### 3C. Consolidate `cli.py` duplicate functions

**File:** `src/parcellate/interfaces/cat12/cli.py`

Remove duplicated `_build_output_path` and `_write_output` from `cli.py`. Import them from `cat12.py` instead. Update `process_single_subject` to pass config to `_write_output`.

### 3D. Integrate into QSIRecon

**File:** `src/parcellate/interfaces/qsirecon/qsirecon.py`

Same pattern as 3B: change `_write_output` signature to accept `config`, call `write_parcellation_sidecar`. Update call site in `_run_recon`.

---

## Implementation Order

1. **Feature 3** (JSON sidecar) — changes `_write_output` signature, foundational for other changes
2. **Feature 1** (runner optimization) — independent of sidecar, touches different functions
3. **Feature 2** (TIV) — builds on the consolidated `_write_output` from Feature 3

---

## Files Modified (Summary)

| File | Changes |
|------|---------|
| `src/parcellate/interfaces/runner.py` | Two-phase execution, cross-atlas parallelism, `as_completed` |
| `src/parcellate/interfaces/utils.py` | Add `write_parcellation_sidecar()` |
| `src/parcellate/interfaces/cat12/loader.py` | Add `discover_cat12_xml()`, `extract_tiv_from_xml()` |
| `src/parcellate/interfaces/cat12/cat12.py` | TIV orchestration, JSON sidecar in `_write_output`, `as_completed` for cross-subject |
| `src/parcellate/interfaces/cat12/cli.py` | Remove duplicate functions, import from `cat12.py`, TIV integration |
| `src/parcellate/interfaces/qsirecon/qsirecon.py` | JSON sidecar in `_write_output`, `as_completed` for cross-subject |
| `tests/test_interfaces_cat12.py` | TIV tests, sidecar tests, updated `_write_output` tests |
| `tests/test_interfaces_qsirecon.py` | Updated `_write_output` tests for sidecar |

---

## Tests

### New tests in `tests/test_interfaces_cat12.py`:
- `test_discover_cat12_xml_finds_files` — create XML in anat dir, verify discovery
- `test_discover_cat12_xml_returns_empty` — no XML files case
- `test_extract_tiv_from_xml_valid` — parse synthetic XML with numeric vol_TIV
- `test_extract_tiv_from_xml_description_text` — non-numeric vol_TIV skipped
- `test_extract_tiv_from_xml_malformed` — returns None for bad XML
- `test_build_tiv_output_path` — verify path structure
- `test_extract_and_write_tiv_creates_file` — end-to-end with tmp_path
- `test_write_parcellation_sidecar_creates_json` — verify JSON content and structure
- `test_write_output_creates_sidecar` — update existing test to verify JSON alongside TSV

### Updated tests:
- `test_write_output_creates_file` (cat12) — pass config to `_write_output`, verify JSON
- `test_run_parcellations_writes_outputs` (cat12) — verify TIV path in outputs
- QSIRecon `_write_output` tests — pass config parameter

---

## Verification

1. `uv run python -m pytest tests/test_interfaces_cat12.py tests/test_interfaces_qsirecon.py -v` — all tests pass
2. `uv run python -m pytest` — full test suite passes
3. `uv run pre-commit run -a` — ruff format + lint clean
4. `uv run mypy src` — type checking passes
