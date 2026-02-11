# Plan 04: Performance Optimizations

## 4.1 Cache Mask Resampling

### Current Behavior

**File:** `parcellation/volume.py:302-304`

```python
if self.mask is not None:
    self._prepared_mask = self._prepare_map(self.mask, ref_img, interpolation="nearest")
    self._prepared_atlas_img = self._apply_mask_to_atlas()
```

Every call to `fit()` resamples the mask image, even when the mask and reference image haven't changed. In a typical workflow, the same mask is used for many scalar maps with the same atlas, so the reference image is always the same.

### Fix

Cache the prepared mask by reference image identity:

```python
def fit(self, scalar_img):
    ...
    ref_img = ...
    self._prepared_atlas_img = self._prepare_map(self.atlas_img, ref_img, interpolation="nearest")

    if self.mask is not None:
        ref_id = id(ref_img)
        if not hasattr(self, '_cached_mask_ref_id') or self._cached_mask_ref_id != ref_id:
            self._prepared_mask = self._prepare_map(self.mask, ref_img, interpolation="nearest")
            self._cached_mask_ref_id = ref_id
        self._prepared_atlas_img = self._apply_mask_to_atlas()

    ...
```

**Impact:** Saves one `resample_to_img` call per `fit()` after the first (significant for large images).

---

## 4.2 Cache Atlas Resampling

### Current Behavior

**File:** `parcellation/volume.py:300`

```python
self._prepared_atlas_img = self._prepare_map(self.atlas_img, ref_img, interpolation="nearest")
```

The atlas is resampled on every `fit()` call, but the atlas never changes within a `VolumetricParcellator` instance.

### Fix

Cache by reference image identity:

```python
if not hasattr(self, '_cached_atlas_ref_id') or self._cached_atlas_ref_id != id(ref_img):
    self._prepared_atlas_img = self._prepare_map(self.atlas_img, ref_img, interpolation="nearest")
    self._cached_atlas_ref_id = id(ref_img)
```

**Impact:** Saves one `resample_to_img` call per `fit()` after the first. Combined with 4.1, this means the second+ `fit()` call on the same atlas only resamples the scalar image.

---

## 4.3 Avoid Redundant `pd.read_csv` for Cached Outputs

### Current Behavior

**Files:**
- `interfaces/cat12/cat12.py:187`
- `interfaces/qsirecon/qsirecon.py:195`

```python
if not config.force and out_path.exists():
    LOGGER.info("Reusing existing parcellation output at %s", out_path)
    _ = pd.read_csv(out_path, sep="\t")  # Read and discard!
    reused_outputs.append(out_path)
```

The CSV is read to validate it exists and is readable, but the result is discarded. For large output files or many atlases, this wastes I/O.

### Fix

Replace with a simple existence check, or a lightweight validation (check file size > 0):

```python
if not config.force and out_path.exists():
    LOGGER.info("Reusing existing parcellation output at %s", out_path)
    reused_outputs.append(out_path)
```

If validation is desired, do it lazily or use `out_path.stat().st_size > 0`.

---

## 4.4 Use `numpy` Vectorized Operations in Stats Loop

### Current Behavior

**File:** `parcellation/volume.py:348-364`

```python
for region_id in self._regions:
    parcel_mask = atlas_data == region_id      # Full array comparison per region
    parcel_values = scalar_data[parcel_mask]    # Array indexing per region
    ...
```

For atlases with many regions (e.g., Schaefer-1000), this performs 1000 full-array comparisons.

### Fix (Future Optimization)

Use `numpy` label-based operations to avoid repeated full-array scans:

```python
from scipy.ndimage import labeled_comprehension
# Or use np.unique with return_inverse:
unique_ids, inverse = np.unique(atlas_data, return_inverse=True)
```

This would allow single-pass computation. However, the current approach is already reasonable for typical atlas sizes (50-400 regions). **Mark as future optimization, not urgent.**

---

## 4.5 Parallel Atlas Processing Within `transform()`

### Current Behavior

`transform()` processes one region at a time sequentially. The `runner.py` parallelizes across scalar maps, but not across regions within a single map.

### Assessment

For typical use cases (50-400 regions, fast numpy operations), the overhead of threading/multiprocessing per region would likely exceed the computation cost. **Not recommended** unless profiling shows this is a bottleneck for very large atlases.

---

## Summary of Recommended Optimizations

| Optimization | Impact | Risk | Recommend |
|-------------|--------|------|-----------|
| 4.1 Cache mask resampling | High (avoids nibabel resample) | Low | Yes |
| 4.2 Cache atlas resampling | High (avoids nibabel resample) | Low | Yes |
| 4.3 Remove redundant CSV read | Low-Medium (I/O savings) | None | Yes |
| 4.4 Vectorized stats loop | Medium (for large atlases) | Medium | Defer |
| 4.5 Parallel region processing | Low | High (overhead) | No |
