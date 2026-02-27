# CLI reference

`parcellate` ships with a BIDS App-compatible command-line interface.

## BIDS App interface (recommended)

```
parcellate <bids_dir> <output_dir> <analysis_level> --pipeline PIPELINE [OPTIONS]
```

### Positional arguments

| Argument | Description |
|----------|-------------|
| `bids_dir` | Root directory of the preprocessing derivatives (CAT12 or QSIRecon output tree). |
| `output_dir` | Destination directory where parcellation results are written. |
| `analysis_level` | Level of analysis. Only `participant` is supported at this time. |

### Required options

| Flag | Description |
|------|-------------|
| `--pipeline {cat12,qsirecon}` | Preprocessing pipeline that produced the input data. |

### Participant / session selection

| Flag | Description |
|------|-------------|
| `--participant-label LABEL [LABEL …]` | Process only these participants (without `sub-` prefix). Default: all. |
| `--session-label ID [ID …]` | Process only these sessions (without `ses-` prefix). Default: all. |

### Configuration

| Flag | Description |
|------|-------------|
| `--config CONFIG.toml` | Path to a TOML file providing atlas definitions and pipeline defaults. |
| `--atlas-config FILE [FILE …]` | One or more TOML files defining atlases (overrides `[[atlases]]` in `--config`). |

### Processing options

| Flag | Default | Description |
|------|---------|-------------|
| `--mask {gm,wm,brain}` / `PATH` | `gm` (CAT12) / none (QSIRecon) | Mask to restrict voxels during parcellation. Accepts built-in MNI152 names or a path to a NIfTI mask. |
| `--mask-threshold FLOAT` | `0.0` | Minimum mask value to include a voxel (strict `>`). Useful with probability maps. |
| `--stat-tier {core,extended,diagnostic,all}` | `diagnostic` | Statistics tier to compute (see [Metrics reference](metrics_reference.md)). |
| `--force` | `False` | Overwrite existing parcellation outputs. |
| `--log-level LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `--n-jobs N` | `1` | Parallel jobs for within-subject parcellation (thread pool). |
| `--n-procs N` | `1` | Parallel processes for across-subject parcellation (process pool). |

### Examples

```bash
# Minimal: one participant, all sessions, default settings
parcellate /data/cat12 /data/parcellations participant \
    --pipeline cat12 \
    --participant-label 01 \
    --config atlases.toml

# Multiple participants with WM mask and extended statistics
parcellate /data/cat12 /data/parcellations participant \
    --pipeline cat12 \
    --participant-label 01 02 03 \
    --session-label ses-01 \
    --mask wm \
    --stat-tier extended \
    --config atlases.toml

# QSIRecon with parallelism
parcellate /data/qsirecon /data/parcellations participant \
    --pipeline qsirecon \
    --config atlases.toml \
    --stat-tier core \
    --n-jobs 4 \
    --n-procs 2
```

---

## Legacy subcommand interface (deprecated)

The old subcommand-based invocation still works but emits a `DeprecationWarning`.
It will be removed in a future major release.

```bash
# Deprecated — use BIDS App interface instead
parcellate cat12 config.toml
parcellate qsirecon config.toml
```

Legacy invocations accept the same optional flags as the BIDS App interface (`--input-root`, `--output-dir`, `--subjects`, `--sessions`, etc.) directly after the subcommand.

---

## TOML configuration file

Atlas definitions and pipeline defaults can be provided in a TOML file:

```toml
input_root = "/data/cat12"
output_dir = "/data/parcellations"
subjects = ["sub-01", "sub-02"]   # optional filter
sessions = ["ses-01"]             # optional filter
mask = "gm"                       # gm | wm | brain | /path/to/mask.nii.gz
mask_threshold = 0.5              # >0 voxels included when using probability maps
stat_tier = "extended"            # core | extended | diagnostic | all
force = false
log_level = "INFO"
n_jobs = 4
n_procs = 2

[[atlases]]
name = "schaefer400"
path = "/path/to/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
lut  = "/path/to/Schaefer2018_400Parcels_7Networks_order_FSLMNI152.tsv"
space = "MNI152NLin2009cAsym"

[[atlases]]
name = "aal"
path = "/path/to/AAL3v1.nii.gz"
lut  = "/path/to/AAL3v1.tsv"
space = "MNI152NLin2009cAsym"
```

CLI flags **always override** values from the TOML file.

---

## `parcellate-cat12` (standalone, deprecated)

The `parcellate-cat12` entry point is a separate CSV-based CLI for CAT12 that
reads configuration from environment variables.  It is deprecated and will be
removed in a future release.  Migrate to `parcellate cat12 config.toml` or
the BIDS App interface.
