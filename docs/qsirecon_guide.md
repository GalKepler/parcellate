# QSIRecon pipeline guide

This guide covers how to use `parcellate` with [QSIRecon](https://qsirecon.readthedocs.io/) dMRI preprocessing outputs.

## Expected input layout

`parcellate` searches for QSIRecon outputs under `bids_dir` following this structure:

```
bids_dir/
└── qsirecon-<workflow>/
    └── sub-<label>/
        └── [ses-<label>/]
            └── dwi/
                └── sub-<label>_[ses-<label>]_space-<space>_model-<model>_param-<param>_dwimap.nii.gz
```

Files are discovered by the `*_dwimap.nii*` glob pattern. BIDS entities
(`space`, `model`, `param`, `desc`) are parsed from the filename automatically.

### Required filename entities

| Entity | Key | Example |
|--------|-----|---------|
| `param` | Diffusion parameter | `FA`, `MD`, `AD`, `RD` |
| `space` | Coordinate space | `MNI152NLin2009cAsym` |

### Optional filename entities

| Entity | Key | Example |
|--------|-----|---------|
| `model` | Reconstruction model | `csd`, `dti` |
| `desc` | Free-form description | `preproc` |

---

## Minimal configuration

```toml
# qsirecon_config.toml
input_root = "/data/qsirecon_derivatives"
output_dir = "/data/parcellations"

[[atlases]]
name = "schaefer400"
path = "/atlases/Schaefer400.nii.gz"
lut  = "/atlases/Schaefer400.tsv"
space = "MNI152NLin2009cAsym"
```

Run:
```bash
parcellate /data/qsirecon_derivatives /data/parcellations participant \
    --pipeline qsirecon \
    --config qsirecon_config.toml
```

---

## Output structure

Results are written under `output_dir/qsirecon-<workflow>/`:

```
output_dir/
└── qsirecon-<workflow>/
    └── sub-<label>/
        └── [ses-<label>/]
            └── dwi/
                └── atlas-<name>/
                    ├── sub-<label>_atlas-<name>_space-<space>_model-<model>_param-FA_parc.tsv
                    ├── sub-<label>_atlas-<name>_space-<space>_model-<model>_param-FA_parc.json
                    └── ...
```

Each `.tsv` file has one row per atlas region with columns:

| Column | Description |
|--------|-------------|
| `index` | Integer region index. |
| `label` | Region name. |
| *stat columns* | One column per statistic in the selected tier. |

---

## Atlas space matching

QSIRecon atlases **must** declare the same `space` entity as the scalar maps.
Unlike the CAT12 interface, there is no default space — if the atlas definition
omits `space`, only scalar maps without a space entity will be matched to it.

**Tip:** Always set `space` in your atlas definition to avoid silent mismatches.

```toml
[[atlases]]
name  = "xtract"
path  = "/atlases/xtract_tracts.nii.gz"   # 4D probabilistic atlas
lut   = "/atlases/xtract_lut.tsv"
space = "MNI152NLin6Asym"
atlas_threshold = 0.25                    # include only high-confidence voxels
```

---

## Probabilistic (4D) atlases

QSIRecon is often used with white-matter tract atlases (e.g. XTRACT) where
each region is encoded as a continuous probability volume in a 4D NIfTI.
Pass the 4D atlas directly — `parcellate` detects the dimensionality automatically.

```toml
[[atlases]]
name            = "xtract"
path            = "/atlases/XTRACT_tracts.nii.gz"   # 4D
lut             = "/atlases/XTRACT.tsv"
space           = "MNI152NLin6Asym"
atlas_threshold = 0.25   # only voxels where tract probability > 0.25
```

---

## Full configuration reference

```toml
input_root = "/data/qsirecon"       # required
output_dir = "/data/parcellations"  # required

# Participant / session filters
subjects = ["sub-01", "sub-02"]     # optional; default = all
sessions = ["ses-01"]               # optional; default = all

# Masking (optional for dMRI; no built-in default)
mask = "/path/to/brain_mask.nii.gz"
mask_threshold = 0.5

# Statistics
stat_tier = "core"                  # core | extended | diagnostic | all

# Execution
force = false
log_level = "INFO"
n_jobs = 4
n_procs = 2

[[atlases]]
name            = "schaefer400"
path            = "/atlases/Schaefer400.nii.gz"
lut             = "/atlases/Schaefer400.tsv"
space           = "MNI152NLin2009cAsym"

[[atlases]]
name            = "xtract"
path            = "/atlases/XTRACT_tracts.nii.gz"
lut             = "/atlases/XTRACT.tsv"
space           = "MNI152NLin6Asym"
atlas_threshold = 0.25
```
