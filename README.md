# parcellate

[![Build status](https://img.shields.io/github/actions/workflow/status/GalKepler/parcellate/main.yml?branch=main)](https://github.com/GalKepler/parcellate/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/GalKepler/parcellate/branch/main/graph/badge.svg)](https://codecov.io/gh/GalKepler/parcellate)
[![Documentation](https://readthedocs.org/projects/neuroparcellate/badge/?version=latest)](https://neuroparcellate.readthedocs.io/en/latest/?version=latest)
[![License](https://img.shields.io/github/license/GalKepler/parcellate)](https://img.shields.io/github/license/GalKepler/parcellate)

> Extract regional statistics from scalar neuroimaging maps using atlas-based parcellation.

## What It Does

**parcellate** is a Python tool that extracts regional statistics from volumetric brain images using atlas-based parcellation. Given a scalar map (e.g., gray matter density, fractional anisotropy) and a labeled atlas, it computes summary statistics for each brain region.

**Key Features:**
- 🧠 **Multiple Pipeline Support**: Integrates with CAT12 (VBM) and QSIRecon (diffusion MRI) outputs
- 📊 **Rich Statistics**: 45 built-in metrics in three tiers (core / extended / diagnostic), including robust statistics, higher-order moments, and normality tests
- ⚡ **Performance**: Parallel processing, smart caching, and optimized resampling
- 📁 **BIDS App**: Follows [BIDS App](https://bids-apps.neuroimaging.io/) positional-argument conventions
- 🔧 **Flexible**: Python API and CLI interfaces, custom atlases and statistics

**Supported Input Formats:**
- **CAT12**: Gray matter (GM), white matter (WM), CSF volumes, cortical thickness maps
- **QSIRecon**: Diffusion scalar maps (FA, MD, AD, RD, etc.)

**Output Format:**
- TSV files with regional statistics (one row per brain region)
- BIDS-derivative compatible directory structure

## Installation

```bash
pip install parcellate
```

For CAT12 CSV batch processing with environment variable support:
```bash
pip install parcellate[dotenv]
```

## Quick Start

### BIDS App interface (recommended)

`parcellate` follows the [BIDS App](https://bids-apps.neuroimaging.io/) convention:

```
parcellate <bids_dir> <output_dir> participant --pipeline {cat12,qsirecon} --config config.toml
```

Create a minimal TOML file with your atlas definitions:

```toml
# config.toml
[[atlases]]
name  = "schaefer400"
path  = "/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
lut   = "/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152.tsv"
space = "MNI152NLin2009cAsym"
```

#### CAT12

```bash
parcellate /data/cat12_derivatives /data/parcellations participant \
    --pipeline cat12 \
    --config config.toml \
    --stat-tier extended
```

#### QSIRecon

```bash
parcellate /data/qsirecon_derivatives /data/parcellations participant \
    --pipeline qsirecon \
    --config config.toml \
    --stat-tier core \
    --n-procs 4
```

#### Common options

| Flag | Description |
|------|-------------|
| `--participant-label 01 02` | Process only these subjects |
| `--session-label ses-01` | Process only this session |
| `--mask gm` / `--mask /path/to/mask.nii.gz` | Brain mask (CAT12 default: `gm`) |
| `--mask-threshold 0.5` | Minimum mask probability |
| `--stat-tier core\|extended\|diagnostic` | Statistics tier (default: `diagnostic`) |
| `--force` | Overwrite existing outputs |
| `--n-jobs N` | Within-subject parallelism |
| `--n-procs N` | Across-subject parallelism |

See the [CLI reference](https://neuroparcellate.readthedocs.io/en/latest/cli_reference.html) for the full list of flags.

### Legacy interface (deprecated)

The old subcommand syntax still works but emits a `DeprecationWarning`:

```bash
# Deprecated — use BIDS App interface above
parcellate cat12 config.toml
parcellate qsirecon config.toml
```

## Configuration Reference

### TOML Configuration

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `input_root` | string | Path to preprocessing derivatives | Required |
| `output_dir` | string | Output directory for parcellations | Required |
| `subjects` | list | Subject IDs to process | All discovered |
| `sessions` | list | Session IDs to process | All discovered |
| `mask` | string | Brain mask path or builtin (`gm`, `wm`, `brain`) | `gm` (CAT12) / none (QSIRecon) |
| `mask_threshold` | float | Minimum mask value to include a voxel | `0.0` |
| `stat_tier` | string | Statistics tier (`core`, `extended`, `diagnostic`, `all`) | `diagnostic` |
| `force` | boolean | Overwrite existing outputs | `false` |
| `log_level` | string | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
| `n_jobs` | integer | Parallel jobs within subject | `1` |
| `n_procs` | integer | Parallel processes across subjects | `1` |

### Atlas Definition

```toml
[[atlases]]
name            = "MyAtlas"                   # Atlas identifier
path            = "/path/to/atlas.nii.gz"     # NIfTI file (3D integer or 4D probabilistic)
lut             = "/path/to/atlas.tsv"        # Optional: TSV with 'index' and 'label' columns
space           = "MNI152NLin2009cAsym"       # Template space
atlas_threshold = 0.0                         # For 4D atlases: minimum voxel probability
```

For the full configuration reference, see the [Configuration reference](https://neuroparcellate.readthedocs.io/en/latest/configuration.html).

## Output Format

Each parcellation produces a TSV file with one row per brain region and one column per statistic. The first two columns are always:

| Column | Description |
|--------|-------------|
| `index` | Integer region index from the atlas |
| `label` | Region name from the lookup table |

Subsequent columns contain the computed statistics. Which columns appear depends on the selected `stat_tier`. For CAT12, a `vol_TIV` column is also appended when XML report files are found.

A JSON sidecar (`.json`) is written alongside each TSV with full provenance (atlas path, mask, thresholds, software version, timestamp).

## Statistics tiers

`parcellate` ships with 45 built-in statistics organized into tiers. Select a tier with `--stat-tier` (CLI) or `stat_tier` (TOML / Python):

| Tier | Count | Typical use |
|------|-------|------------|
| `core` | 6 | Fast exploration, large cohorts |
| `extended` | 21 | Production pipelines |
| `diagnostic` (default) | 45 | QC, distribution inspection |

**Core statistics:** `mean`, `std`, `median`, `volume_mm3`, `voxel_count`, `sum`

**Extended adds:** robust means (MAD, z-score, IQR filtered), dispersion (`cv`, `robust_cv`), shape (`skewness`, `excess_kurtosis`), key percentiles (5th, 25th, 75th, 95th)

**Diagnostic adds:** normality tests (Shapiro-Wilk, D'Agostino K²), outlier proportions, tail mass, entropy, boolean QC flags

See the full [Metrics reference](https://neuroparcellate.readthedocs.io/en/latest/metrics_reference.html) for descriptions of all 45 statistics.

## Python API

Use the core parcellation engine directly:

```python
from parcellate import VolumetricParcellator
import nibabel as nib

# Initialize parcellator
parcellator = VolumetricParcellator(
    atlas_img="atlas.nii.gz",
    lut="atlas.tsv",
    mask="gm",                  # built-in MNI152 grey-matter mask
    stat_tier="extended",       # 21 statistics (default: all 45)
    resampling_target="data",   # resample atlas to scalar-map space
)

# Fit once, then transform one or more maps
parcellator.fit("gm_map.nii.gz")
stats = parcellator.transform("gm_map.nii.gz")
print(stats.head())
```

### Custom statistics

Supply a dict of callables to override the built-in statistics entirely:

```python
import numpy as np

parcellator = VolumetricParcellator(
    atlas_img="atlas.nii.gz",
    stat_functions={
        "iqr": lambda x: float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
        "q90": lambda x: float(np.nanpercentile(x, 90)),
    },
)
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/GalKepler/parcellate.git
cd parcellate

# Install development dependencies
make install

# Run tests
make test

# Run code quality checks
make check
```

### Running Tests

```bash
# All tests
uv run python -m pytest

# Specific test file
uv run python -m pytest tests/test_parcellator.py

# With coverage
uv run python -m pytest --cov=src/parcellate --cov-report=html
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks (`make check && make test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Documentation

Full documentation is available at [https://GalKepler.github.io/parcellate/](https://GalKepler.github.io/parcellate/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{parcellate,
  author = {Kepler, Gal},
  title = {parcellate: Atlas-based parcellation of neuroimaging data},
  url = {https://github.com/GalKepler/parcellate},
  year = {2024}
}
```

## Acknowledgments

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
