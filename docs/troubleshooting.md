# Troubleshooting

This page covers common errors, warnings, and their solutions.

---

## CLI errors

### `error: the following arguments are required: --pipeline`

**Cause:** The BIDS App interface requires `--pipeline` to be specified.

**Fix:**
```bash
# Add --pipeline {cat12,qsirecon}
parcellate /data/cat12 /data/out participant --pipeline cat12 --config atlases.toml
```

---

### `error: argument analysis_level: invalid choice: 'group'`

**Cause:** Only `participant`-level analysis is supported.

**Fix:** Use `participant` as the analysis level:
```bash
parcellate /data/cat12 /data/out participant --pipeline cat12 ...
```

---

### `DeprecationWarning: 'parcellate cat12' is deprecated`

**Cause:** You are using the old subcommand syntax (`parcellate cat12 config.toml`).

**Fix:** Migrate to the BIDS App interface:
```bash
# Old (deprecated)
parcellate cat12 config.toml

# New
parcellate /data/cat12 /data/out participant --pipeline cat12 --config config.toml
```

---

### `DeprecationWarning: The 'parcellate-cat12' command is deprecated`

**Cause:** You are using the CSV-based `parcellate-cat12` entry point.

**Fix:** Migrate to the BIDS App interface. See the [CAT12 guide](cat12_guide.md) for the equivalent configuration.

---

## Atlas errors

### `ValueError: Unknown stat_tier "…". Valid tiers: "core", "extended", "diagnostic", "all".`

**Cause:** An unrecognized value was passed to `--stat-tier` (CLI) or `stat_tier` (TOML/Python).

**Fix:** Use one of the valid tiers:
```bash
--stat-tier core      # 6 statistics
--stat-tier extended  # 21 statistics
--stat-tier diagnostic  # all 45 (default)
```

---

### `MissingStatisticalFunctionError`

**Cause:** `stat_functions` was provided but is empty, and no fallback is available.

**Fix:** Provide at least one callable in `stat_functions`, or remove the argument to use the default tier.

---

### `ValueError: Atlas space "…" not found in any scalar map`

**Cause:** The atlas `space` entry in the config does not match the `space-` entity in any discovered scalar map filenames.

**Fix:** Check that the `space` value in the atlas definition exactly matches the space encoded in the input filenames. For CAT12, the default space is `MNI152NLin2009cAsym`. For QSIRecon, the space is embedded in the `*_dwimap.nii.gz` filename.

---

### `MissingLUTColumnsError`

**Cause:** The lookup table TSV does not have the required `index` and/or `label` columns.

**Fix:** Ensure your LUT has at minimum:
```
index   label
1       Left-Frontal-Pole
2       Right-Frontal-Pole
```

---

### `ParcellatorNotFittedError`

**Cause:** `transform()` was called before `fit()`.

**Fix:**
```python
parcellator.fit("subject_T1w.nii.gz")
stats = parcellator.transform("subject_T1w.nii.gz")
```

---

## Input discovery errors

### No subjects found / empty run

**Cause:** `parcellate` could not find any valid input files under `bids_dir`.

**Checklist:**
1. Verify the directory structure matches the expected layout (see [CAT12 guide](cat12_guide.md) or [QSIRecon guide](qsirecon_guide.md)).
2. For CAT12: check that `mwp1*.nii.gz` / `mwp2*.nii.gz` files exist under `sub-*/anat/`.
3. For QSIRecon: check that `*_dwimap.nii.gz` files exist under `qsirecon-*/sub-*/dwi/`.
4. Confirm `--participant-label` values match directory names (without the `sub-` prefix).

---

### `FileNotFoundError` for atlas or mask path

**Cause:** A path specified in the config or CLI does not exist.

**Fix:** Use absolute paths. Relative paths are resolved from the current working directory, which may differ depending on how `parcellate` is invoked.

---

## Resampling / space errors

### Outputs contain unexpected NaN values

**Cause:** The atlas and scalar map spaces do not align after resampling.

**Checklist:**
1. Confirm `space` in the atlas definition matches the space of the scalar map.
2. If using a custom mask, ensure it is in the same space as the scalar maps.
3. Check `atlas_threshold` — if set too high, all voxels in a region may be excluded, producing NaN statistics.

---

### Slow parcellation / high memory usage

**Cause:** The default `resampling_target="data"` resamples the atlas into scalar-map space. If the scalar map is very high resolution, this creates large in-memory arrays.

**Fix (Python API):**
```python
parcellator = VolumetricParcellator(
    atlas_img="atlas.nii.gz",
    resampling_target="labels",  # resample scalar maps to atlas resolution
)
```

**Fix (TOML):** `resampling_target` is not yet exposed as a TOML key. Use `n_jobs` / `n_procs` to manage resource usage.

---

## Parallelism

### `BrokenProcessPool` / worker crash

**Cause:** A subprocess died unexpectedly, often due to running out of memory.

**Fix:** Reduce `n_procs` (across-subject processes) or `n_jobs` (within-subject threads):
```toml
n_jobs  = 2
n_procs = 1
```

---

## Getting help

If none of the above resolves your issue, please open a GitHub issue at:

```
https://github.com/GalKepler/parcellate/issues
```

Include:
- The full error traceback
- The TOML config (redact sensitive paths)
- The output of `parcellate --version`
- The Python and OS versions (`python --version`, `uname -a`)
