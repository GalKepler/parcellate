# Plan 03: Bugs and Correctness

## 3.1 [BUG] QSIRecon Loader Only Returns First Session

### Severity: HIGH

**File:** `interfaces/qsirecon/loader.py:45-74`

**Problem:** The `_process_subject()` function loops over sessions but returns on the **first successful session**, discarding all others:

```python
def _process_subject(root, subject_id, atlas_definitions):
    sessions = _discover_sessions(root, subject_id)
    for session_id in sessions:
        try:
            context = SubjectContext(subject_id=subject_id, session_id=session_id)
            scalar_maps = discover_scalar_maps(root=root, subject=subject_id, session=session_id)
            if not scalar_maps:
                return None      # <-- Returns None, skips remaining sessions
            return ReconInput(   # <-- Returns on FIRST session, skips rest
                context=context,
                scalar_maps=scalar_maps,
                atlases=atlas_definitions,
            )
        except Exception:
            return None          # <-- Returns None on error, skips rest
```

**Expected behavior:** Should return a `ReconInput` for **each** session with scalar maps.

### Fix

Change return type to `list[ReconInput]` and accumulate results:

```python
def _process_subject(root, subject_id, atlas_definitions) -> list[ReconInput]:
    sessions = _discover_sessions(root, subject_id)
    results = []
    for session_id in sessions:
        try:
            context = SubjectContext(subject_id=subject_id, session_id=session_id)
            scalar_maps = discover_scalar_maps(root=root, subject=subject_id, session=session_id)
            if not scalar_maps:
                logger.debug(f"No scalar maps for sub-{subject_id} ses-{session_id}")
                continue  # Try next session
            results.append(ReconInput(
                context=context,
                scalar_maps=scalar_maps,
                atlases=atlas_definitions,
            ))
        except Exception:
            logger.exception(f"Error processing sub-{subject_id} ses-{session_id}")
            continue  # Try next session
    return results
```

Update `load_qsirecon_inputs()` to flatten the results from each subject.

---

## 3.2 [BUG] `voxel_count()` Counts Non-zero Values, Not All Voxels

### Severity: LOW (semantic, not functional)

**File:** `metrics/volume.py:35-52`

**Problem:** The function name and docstring say "count voxels in a parcel" but the implementation counts only non-zero, non-NaN voxels:

```python
def voxel_count(parcel_values: np.ndarray) -> int:
    num_voxels = np.sum(parcel_values.astype(bool))  # Excludes zeros
    return int(num_voxels)
```

Since `parcel_values` is already filtered to only include voxels within the parcel mask (`scalar_data[parcel_mask]` in volume.py:353), zeros are legitimate scalar values that should be counted.

### Fix Options

**Option A** (preserve current behavior, fix naming): Rename to `nonzero_voxel_count` and update docstring.

**Option B** (fix behavior): Use `len(parcel_values)` or `np.sum(~np.isnan(parcel_values))` if NaN exclusion is desired.

**Recommendation:** Option B — count all non-NaN voxels:

```python
def voxel_count(parcel_values: np.ndarray) -> int:
    return int(np.sum(~np.isnan(parcel_values)))
```

---

## 3.3 Skewness/Kurtosis Not Wired Into BUILTIN_STATISTICS

### Severity: LOW

**File:** `metrics/volume.py:283-330`

**Problem:** `skewness()` and `kurtosis()` functions are defined (lines 283-312) and `scipy.stats` is imported (line 7), but they are **not included** in the `BUILTIN_STATISTICS` list (lines 316-330).

### Fix

Add to `BUILTIN_STATISTICS`:

```python
BUILTIN_STATISTICS: list[Statistic] = [
    # ... existing entries ...
    Statistic(name="skewness", function=skewness),
    Statistic(name="kurtosis", function=kurtosis),
]
```

And add `scipy` as an explicit dependency in `pyproject.toml`.

---

## 3.4 f-string Logging (Style/Correctness)

### Severity: LOW

**Files with f-string logging:**
- `interfaces/qsirecon/loader.py:60-61, 63-64, 73, 89-90, 191`
- `interfaces/cat12/loader.py:43, 52, 91`
- `interfaces/qsirecon/qsirecon.py:242-243, 254`

**Problem:** Using f-strings with `logger.info(f"...")` evaluates the string even when the log level would suppress the message. The idiomatic pattern is `logger.info("...", arg1, arg2)`.

### Fix

Replace all f-string logging calls with `%s`-style formatting:

```python
# Before
logger.info(f"Discovered {len(subj_list)} subjects")

# After
logger.info("Discovered %d subjects", len(subj_list))
```

---

## 3.5 `ScalarMapDefinition` TYPE_CHECKING Hack

### Severity: MEDIUM

**File:** `interfaces/models.py:12-15`

```python
if TYPE_CHECKING:
    ScalarMapDefinition = Any
else:
    ScalarMapDefinition = "ScalarMapDefinition"
```

**Problem:** At runtime, `ScalarMapDefinition` is a string literal `"ScalarMapDefinition"`, not a type. This defeats any runtime type checking and makes `isinstance()` impossible. At type-check time, it's `Any`, which defeats static analysis too.

### Fix

See Plan 02, section 2.2 — introduce `ScalarMapBase` as a concrete shared base class.

---

## 3.6 Bare Exception Handling in Loaders

### Severity: LOW

**Files:**
- `interfaces/qsirecon/loader.py:72` — `except Exception:`
- `interfaces/cat12/loader.py:51` — `except Exception:`
- `interfaces/runner.py:84` — `except Exception:`

**Problem:** Catching `Exception` is too broad. Specific exceptions (e.g., `FileNotFoundError`, `ValueError`) should be caught to avoid masking programming errors.

### Fix

Narrow to specific expected exceptions:

```python
except (FileNotFoundError, ValueError, OSError) as e:
    logger.warning("Failed to process sub-%s: %s", subject_id, e)
    continue
```

Or at minimum, keep `Exception` but add a comment explaining why it's intentional (defensive processing in batch pipelines).
