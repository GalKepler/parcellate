# Configuration reference

`parcellate` is configured through TOML files. CLI flags always override values from the TOML file.

## File structure

A configuration file can contain top-level keys (pipeline settings) and one or more `[[atlases]]` sections:

```toml
# Pipeline settings
input_root = "/data/derivatives"
output_dir  = "/data/parcellations"

[[atlases]]
name  = "schaefer400"
path  = "/atlases/Schaefer400.nii.gz"
lut   = "/atlases/Schaefer400.tsv"
space = "MNI152NLin2009cAsym"
```

Pass the file with `--config`:

```bash
parcellate /data/cat12 /data/out participant --pipeline cat12 --config my_config.toml
```

---

## Top-level settings

### Paths

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `input_root` | string | Yes | Root directory of preprocessing derivatives. Overridden by the `bids_dir` positional argument on the CLI. |
| `output_dir` | string | Yes | Destination directory for parcellation outputs. |

### Participant / session filters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `subjects` | list of strings | all | Subject IDs to process (with or without `sub-` prefix). |
| `sessions` | list of strings | all | Session IDs to process (with or without `ses-` prefix). |

```toml
subjects = ["sub-01", "sub-02"]
sessions = ["ses-baseline"]
```

### Masking

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mask` | string | `"gm"` (CAT12) / none (QSIRecon) | Brain mask. Accepts built-in names (`gm`, `wm`, `brain`) or an absolute path to a NIfTI file. |
| `mask_threshold` | float | `0.0` | Minimum mask value to include a voxel (strict `>`). Useful with probabilistic masks. |

```toml
# Built-in grey-matter mask (CAT12 default)
mask = "gm"
mask_threshold = 0.0

# Custom probabilistic mask at 50% threshold
mask = "/path/to/custom_gm.nii.gz"
mask_threshold = 0.5
```

### Statistics

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `stat_tier` | string | `"diagnostic"` | Tier of statistics to compute. One of `core`, `extended`, `diagnostic`, `all`. See [Metrics reference](metrics_reference.md). |

```toml
stat_tier = "extended"   # 21 statistics — good balance for production runs
```

### Execution

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `force` | boolean | `false` | Overwrite existing output files. When `false`, existing files are skipped. |
| `log_level` | string | `"INFO"` | Logging verbosity. One of `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `n_jobs` | integer | `1` | Number of parallel threads for within-subject parcellation. |
| `n_procs` | integer | `1` | Number of parallel processes for across-subject parcellation. |

```toml
force     = false
log_level = "INFO"
n_jobs    = 4
n_procs   = 2
```

---

## Atlas definitions (`[[atlases]]`)

Each `[[atlases]]` block defines one atlas. Multiple atlases can be stacked in a single file.

### Required keys

| Key | Type | Description |
|-----|------|-------------|
| `name` | string | Short identifier embedded in output filenames (`atlas-<name>`). |
| `path` | string | Absolute path to the atlas NIfTI file (3D integer-labeled or 4D probabilistic). |

### Optional keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lut` | string | — | Path to a lookup table TSV with `index` and `label` columns. |
| `space` | string | — | BIDS space entity (`MNI152NLin2009cAsym`, `MNI152NLin6Asym`, …). Must match the scalar map space. |
| `resolution` | string | — | Optional BIDS `res-` entity (e.g., `"1mm"`). Informational only. |
| `atlas_threshold` | float | `0.0` | For 4D probabilistic atlases: minimum voxel probability to include (strict `>`). |

```toml
[[atlases]]
name            = "schaefer400"
path            = "/atlases/Schaefer400.nii.gz"
lut             = "/atlases/Schaefer400.tsv"
space           = "MNI152NLin2009cAsym"

[[atlases]]
name            = "xtract"
path            = "/atlases/XTRACT_tracts.nii.gz"   # 4D probabilistic
lut             = "/atlases/XTRACT.tsv"
space           = "MNI152NLin6Asym"
atlas_threshold = 0.25
```

### Lookup table format

The LUT file is a tab-separated file with at minimum these two columns:

| Column | Description |
|--------|-------------|
| `index` | Integer label value in the atlas NIfTI. |
| `label` | Human-readable region name. |

Additional columns are allowed and preserved in the output TSV.

Example `atlas.tsv`:
```
index	label
1	Left-Frontal-Pole
2	Right-Frontal-Pole
3	Left-Superior-Frontal-Gyrus
```

---

## Providing atlases separately (`--atlas-config`)

Atlas definitions can be split into a separate TOML file and passed with `--atlas-config`. This is convenient when reusing the same atlas set across multiple pipeline runs.

```toml
# atlases.toml — atlas definitions only
[[atlases]]
name  = "schaefer400"
path  = "/atlases/Schaefer400.nii.gz"
lut   = "/atlases/Schaefer400.tsv"
space = "MNI152NLin2009cAsym"
```

```bash
parcellate /data/cat12 /data/out participant \
    --pipeline cat12 \
    --config pipeline.toml \
    --atlas-config atlases.toml
```

When `--atlas-config` is supplied, it **overrides** the `[[atlases]]` sections in `--config`.

---

## Complete example

```toml
# full_config.toml

input_root = "/data/cat12"
output_dir = "/data/parcellations"

subjects = ["sub-01", "sub-02", "sub-03"]
sessions = ["ses-baseline"]

mask           = "gm"
mask_threshold = 0.0

stat_tier = "extended"

force     = false
log_level = "INFO"
n_jobs    = 4
n_procs   = 2

[[atlases]]
name  = "schaefer400"
path  = "/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
lut   = "/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152.tsv"
space = "MNI152NLin2009cAsym"

[[atlases]]
name  = "aal3"
path  = "/atlases/AAL3v1.nii.gz"
lut   = "/atlases/AAL3v1.tsv"
space = "MNI152NLin2009cAsym"
```

Run:
```bash
parcellate /data/cat12 /data/parcellations participant \
    --pipeline cat12 \
    --config full_config.toml
```
