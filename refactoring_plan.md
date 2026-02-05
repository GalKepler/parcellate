# Plan: Optimize Parcellation Pipeline + Bug Fixes + Refactoring

## Overview

Four phases, ordered by risk (lowest first). Each phase is independently mergeable.

---

## Phase 1: Fix `voxel_count()` parameter naming and docs

**File:** `src/parcellate/metrics/volume.py` (lines 35-49)

**Problem:** Parameter is named `parcel_mask` (suggesting a boolean mask) but actually receives `parcel_values` (float scalar values extracted via `scalar_data[atlas == region_id]`). The docstring says "A boolean mask of the parcel" which is wrong.

The current implementation `np.sum(parcel_mask.astype(bool))` counts non-zero-or-NaN values. For typical neuroimaging data this equals region size, but the semantics are accidental.

**Change:** Rename parameter to `parcel_values`, update docstring to document actual behavior. Keep `np.sum(...astype(bool))` logic unchanged to avoid breaking existing integration tests.

**Test file:** `tests/test_metrics.py` — no changes needed (existing tests pass).

---

## Phase 2: Cache fitted scalar image to avoid redundant resampling

**File:** `src/parcellate/parcellation/volume.py`

**Problem (HIGH impact):** The runner calls `vp.fit(scalar_maps[0].nifti_path)` then immediately `vp.transform(scalar_maps[0].nifti_path)`. Each `transform()` call unconditionally resamples the scalar image via `nilearn.image.resample_to_img(force_resample=True)` (line 317-321), even when the image was already resampled in `fit()`. Resampling is the most expensive operation in the pipeline.

**Change in `fit()` (~line 292):**
- Store the identity of the fitted scalar image as `self._fitted_scalar_id` (string path or object `id()`).

**Change in `transform()` (~line 317):**
- Compare the incoming image identity against `self._fitted_scalar_id`.
- If same, reuse `self._prepared_scalar_img` from `fit()`.
- If different, resample as before.

**Path comparison:** Use `str(path)` for `str`/`Path` inputs, `id(img)` for in-memory `Nifti1Image` objects.

**Test additions in `tests/test_parcellator.py`:**
- Test that `transform()` with the same path as `fit()` produces correct results.
- Test that `transform()` with a different path still resamples correctly.

---

## Phase 3: Optimize stats loop with dict-based accumulation

**File:** `src/parcellate/parcellation/volume.py` (lines 325-344)

**Problem (MEDIUM impact):** Two inefficiencies in the inner loop:
1. `result["index"].values` (line 331) creates a new array every iteration — O(N) per region.
2. `result.loc[result["index"] == region_id, stat_name] = stat_value` (line 343) does a linear scan for every stat assignment. With 13 stats x 100 regions = 1,300 linear scans.

**Change:** Replace the loop with dict-based accumulation:
1. Pre-compute `valid_region_ids = set(result["index"])` once.
2. Accumulate stats in a `dict[int, dict[str, float]]` keyed by region_id.
3. Build a `pd.DataFrame` from the dict and merge into `result` with a single `merge()` call.

This reduces complexity from O(R x S x N) to O(R x S + N).

**Test impact:** All existing tests pass unchanged — same output, different construction path.

---

## Phase 4: Extract shared code from cat12/qsirecon interfaces

**Problem:** The `runner.py`, `planner.py`, and `models.py` files are 100% duplicated between `interfaces/cat12/` and `interfaces/qsirecon/`. Helper functions `_parse_log_level()` and `_as_list()` are duplicated in `cat12.py` and `qsirecon.py`.

### New files to create

| File | Contains |
|------|----------|
| `src/parcellate/interfaces/models.py` | `SubjectContext`, `AtlasDefinition`, `ReconInput`, `ParcellationOutput` |
| `src/parcellate/interfaces/planner.py` | `_space_match()`, `plan_parcellation_workflow()` |
| `src/parcellate/interfaces/runner.py` | `ScalarMapSpaceMismatchError`, `_validate_scalar_map_spaces()`, `run_parcellation_workflow()` |
| `src/parcellate/interfaces/utils.py` | `_parse_log_level()`, `_as_list()` |

### Files to update (import from shared, keep backward-compat re-exports)

| File | Change |
|------|--------|
| `interfaces/cat12/models.py` | Import shared dataclasses from `interfaces.models`, re-export. Keep `TissueType`, `ScalarMapDefinition`, `Cat12Config`. |
| `interfaces/qsirecon/models.py` | Import shared dataclasses from `interfaces.models`, re-export. Keep `ScalarMapDefinition`, `QSIReconConfig`. |
| `interfaces/cat12/planner.py` | Import from `interfaces.planner`, alias as `plan_cat12_parcellation_workflow`. |
| `interfaces/qsirecon/planner.py` | Import from `interfaces.planner`, alias as `plan_qsirecon_parcellation_workflow`. |
| `interfaces/cat12/runner.py` | Import from `interfaces.runner`, alias as `run_cat12_parcellation_workflow`. |
| `interfaces/qsirecon/runner.py` | Import from `interfaces.runner`, alias as `run_qsirecon_parcellation_workflow`. |
| `interfaces/cat12/cat12.py` | Import `_parse_log_level`, `_as_list` from `interfaces.utils`. |
| `interfaces/qsirecon/qsirecon.py` | Import `_parse_log_level`, `_as_list` from `interfaces.utils`. |

### What stays interface-specific (NOT extracted)

- `ScalarMapDefinition` — different fields per interface
- Config classes — `Cat12Config` has `atlases`, `QSIReconConfig` does not
- `loader.py` — fundamentally different discovery logic
- `_build_output_path()` — different directory structure and filename entities
- `cli.py` — CAT12-specific CSV+parallel CLI

### Typing for shared runner

The shared `run_parcellation_workflow()` needs to accept both `Cat12Config` and `QSIReconConfig`. Use duck typing — both have `mask`, `background_label`, and `resampling_target` attributes. Alternatively, define a `ParcellationConfig` Protocol in `interfaces/models.py`.

### Backward compatibility

All existing imports like `from parcellate.interfaces.cat12.models import SubjectContext` continue to work because the per-interface modules re-export from the shared module. No consumer code needs changes.

---

## Verification

After each phase, run:
```bash
python -m pytest tests/ -v
```

After Phase 4, also verify:
```bash
python -c "from parcellate.interfaces.cat12.models import SubjectContext, AtlasDefinition"
python -c "from parcellate.interfaces.qsirecon.models import SubjectContext, AtlasDefinition"
python -c "from parcellate.interfaces.cat12.runner import run_cat12_parcellation_workflow"
python -c "from parcellate.interfaces.qsirecon.planner import plan_qsirecon_parcellation_workflow"
parcellate --help
parcellate cat12 --help
parcellate qsirecon --help
```
