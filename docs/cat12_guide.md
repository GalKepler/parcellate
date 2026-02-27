# CAT12 pipeline guide

This guide covers how to use `parcellate` with [CAT12](https://neuro-jena.github.io/cat/) preprocessing outputs.

## Expected input layout

`parcellate` searches for CAT12 outputs under `bids_dir` following this structure:

```
bids_dir/
└── sub-<label>/
    └── [ses-<label>/]
        └── anat/
            ├── mwp1<basename>.nii.gz   # gray matter (GM) modulated
            ├── mwp2<basename>.nii.gz   # white matter (WM) modulated
            ├── wct<basename>.nii.gz    # cortical thickness
            └── cat_<basename>.xml      # CAT12 report (optional; for TIV)
```

The subdirectory `anat/` can be nested further (e.g. inside a `CAT12/` folder)
— the loader searches recursively.

### Tissue types

| File pattern | Tissue type | Entity in output |
|--------------|-------------|-----------------|
| `mwp1*` | Gray matter | `tissue-GM` |
| `mwp2*` | White matter | `tissue-WM` |
| `wct*` | Cortical thickness | `tissue-CT` |

---

## Minimal configuration

```toml
# cat12_config.toml
input_root = "/data/cat12_derivatives"
output_dir = "/data/parcellations"

[[atlases]]
name = "schaefer400"
path = "/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
lut  = "/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152.tsv"
space = "MNI152NLin2009cAsym"
```

Run:
```bash
parcellate /data/cat12_derivatives /data/parcellations participant \
    --pipeline cat12 \
    --config cat12_config.toml
```

---

## Output structure

Results are written under `output_dir/cat12/`:

```
output_dir/
└── cat12/
    └── sub-<label>/
        └── [ses-<label>/]
            └── anat/
                └── atlas-<name>/
                    ├── sub-<label>_[ses-<label>]_atlas-<name>_space-<space>_tissue-GM_mask-gm_parc.tsv
                    ├── sub-<label>_[ses-<label>]_atlas-<name>_space-<space>_tissue-GM_mask-gm_parc.json
                    ├── sub-<label>_[ses-<label>]_atlas-<name>_space-<space>_tissue-WM_mask-gm_parc.tsv
                    ├── ...
                    └── sub-<label>_[ses-<label>]_tiv.tsv    # if XML report found
```

Each `.tsv` file contains one row per atlas region with columns:

| Column | Description |
|--------|-------------|
| `index` | Integer region index (from the lookup table). |
| `label` | Region name (from the lookup table). |
| `vol_TIV` | Total Intracranial Volume in cm³ (if XML report available). |
| *stat columns* | One column per statistic in the selected tier. |

Each `.json` sidecar records provenance: original file, atlas, mask,
thresholds, software version, and timestamp.

---

## Masking

CAT12 defaults to the built-in MNI152 gray-matter mask (`mask = "gm"`).
Override with `--mask`:

```bash
# Use white matter mask instead
parcellate /bids /out participant --pipeline cat12 --mask wm --config cfg.toml

# Use a custom probability map with 50% threshold
parcellate /bids /out participant --pipeline cat12 \
    --mask /path/to/custom_gm.nii.gz \
    --mask-threshold 0.5 \
    --config cfg.toml
```

---

## TIV extraction

When CAT12 XML report files (`cat_*.xml`) are found alongside the NIfTI outputs,
`parcellate` automatically extracts the Total Intracranial Volume and:

1. Appends a `vol_TIV` column to each parcellation TSV.
2. Writes a standalone `<context>_tiv.tsv` file.

No extra configuration is required.

---

## Full configuration reference

```toml
input_root = "/data/cat12"          # required
output_dir = "/data/parcellations"  # required

# Participant / session filters
subjects = ["sub-01", "sub-02"]     # optional; default = all
sessions = ["ses-01"]               # optional; default = all

# Masking
mask = "gm"                         # gm | wm | brain | /path/to/mask.nii.gz
mask_threshold = 0.0                # float; voxels with mask > threshold are included

# Statistics
stat_tier = "extended"              # core | extended | diagnostic | all

# Execution
force = false                       # overwrite existing outputs
log_level = "INFO"                  # DEBUG | INFO | WARNING | ERROR
n_jobs = 4                          # within-subject parallelism (threads)
n_procs = 2                         # across-subject parallelism (processes)

[[atlases]]
name  = "schaefer400"
path  = "/atlases/Schaefer400.nii.gz"
lut   = "/atlases/Schaefer400.tsv"
space = "MNI152NLin2009cAsym"       # must match CAT12 output space
# resolution = "1mm"               # optional BIDS entity (res-*)
# atlas_threshold = 0.0            # for 4D probabilistic atlases
```
