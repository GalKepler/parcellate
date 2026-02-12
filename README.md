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
- 📊 **Rich Statistics**: 13+ built-in metrics including mean, median, volume, robust statistics, and higher-order moments
- ⚡ **Performance**: Parallel processing, smart caching, and optimized resampling
- 📁 **BIDS-Compatible**: Outputs follow BIDS-derivative conventions
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

### CAT12 Pipeline

Process CAT12 VBM outputs with a TOML configuration file:

```bash
parcellate cat12 config.toml
```

**Example config.toml:**
```toml
input_root = "/data/cat12_derivatives"
output_dir = "/data/parcellations"
subjects = ["sub-01", "sub-02"]  # Optional: process specific subjects
force = false                     # Skip existing outputs
log_level = "INFO"
n_jobs = 4                        # Parallel jobs within subject
n_procs = 2                       # Parallel processes across subjects

[[atlases]]
name = "Schaefer400"
path = "/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
lut = "/atlases/Schaefer2018_400Parcels_7Networks_order.tsv"
space = "MNI152NLin2009cAsym"
```

**Example output:**
```
parcellations/
└── sub-01/
    └── anat/
        ├── sub-01_space-MNI152NLin2009cAsym_atlas-Schaefer400_desc-gm_stats.tsv
        ├── sub-01_space-MNI152NLin2009cAsym_atlas-Schaefer400_desc-wm_stats.tsv
        └── sub-01_space-MNI152NLin2009cAsym_atlas-Schaefer400_desc-ct_stats.tsv
```

### QSIRecon Pipeline

Process QSIRecon diffusion outputs:

```bash
parcellate qsirecon --config config.toml --input-root /data/qsirecon
```

**Example config.toml:**
```toml
input_root = "/data/qsirecon_derivatives"
output_dir = "/data/parcellations"
sessions = ["ses-01"]
n_jobs = 4

[[atlases]]
name = "JHU"
path = "/atlases/JHU-ICBM-labels-1mm.nii.gz"
lut = "/atlases/JHU-ICBM-labels-1mm.tsv"
space = "MNI152NLin6Asym"
```

### CAT12 CSV Mode (Batch Processing)

Process multiple subjects from a CSV file:

```bash
parcellate-cat12 subjects.csv --root /data/cat12 --atlas-path /atlases/schaefer400.nii.gz
```

**Example subjects.csv:**
```csv
subject_id,session_id
sub-01,ses-baseline
sub-02,ses-baseline
```

Environment variables can be used for configuration:
```bash
export CAT12_ROOT=/data/cat12_derivatives
export CAT12_OUTPUT_DIR=/data/parcellations
export CAT12_ATLAS_PATHS=/atlases/atlas1.nii.gz,/atlases/atlas2.nii.gz
export CAT12_ATLAS_NAMES=Schaefer400,AAL3

parcellate-cat12 subjects.csv
```

## Configuration Reference

### TOML Configuration

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `input_root` | string | Path to preprocessing derivatives | Required |
| `output_dir` | string | Output directory for parcellations | `{input_root}/parcellations` |
| `subjects` | list | Subject IDs to process | All discovered |
| `sessions` | list | Session IDs to process | All discovered |
| `mask` | string | Brain mask path or builtin (`gm`, `wm`, `brain`) | None |
| `force` | boolean | Overwrite existing outputs | `false` |
| `log_level` | string | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) | `INFO` |
| `n_jobs` | integer | Parallel jobs within subject | `1` |
| `n_procs` | integer | Parallel processes across subjects | `1` |

### Atlas Definition

```toml
[[atlases]]
name = "MyAtlas"           # Atlas identifier
path = "/path/to/atlas.nii.gz"  # NIfTI file with integer labels
lut = "/path/to/atlas.tsv"      # Optional: TSV with columns 'index' and 'label'
space = "MNI152NLin2009cAsym"   # Template space
```

The LUT (lookup table) TSV should have:
- `index`: Integer region IDs matching atlas labels
- `label`: Region names

### Environment Variables (CAT12 CSV Mode)

| Variable | Description |
|----------|-------------|
| `CAT12_ROOT` | Input root directory (required) |
| `CAT12_OUTPUT_DIR` | Output directory |
| `CAT12_ATLAS_PATHS` | Comma-separated atlas paths |
| `CAT12_ATLAS_NAMES` | Comma-separated atlas names |
| `CAT12_ATLAS_SPACE` | Space for all atlases |
| `CAT12_MASK` | Mask path or builtin name |
| `CAT12_LOG_LEVEL` | Logging level |

## Output Format

Each parcellation produces a TSV file with one row per brain region:

| Column | Description |
|--------|-------------|
| `index` | Region ID from atlas |
| `label` | Region name (if LUT provided) |
| `mean` | Mean intensity |
| `std` | Standard deviation |
| `median` | Median intensity |
| `mad_median` | Median absolute deviation |
| `min` | Minimum intensity |
| `max` | Maximum intensity |
| `range` | Max - Min |
| `volume` | Sum of intensities |
| `voxel_count` | Number of voxels in region |
| `z_filtered_mean` | Mean after removing outliers (|z| > 3) |
| `z_filtered_std` | Std after removing outliers |
| `skewness` | Distribution skewness |
| `kurtosis` | Distribution kurtosis |

## Available Statistics

| Statistic | Description | Edge Case Behavior |
|-----------|-------------|-------------------|
| `mean` | Arithmetic mean | NaN for empty regions |
| `std` | Standard deviation | 0 for constant values |
| `median` | 50th percentile | NaN for empty regions |
| `mad_median` | Median absolute deviation | Robust alternative to std |
| `min` / `max` | Extreme values | NaN for empty regions |
| `range` | Max - Min | 0 for constant values |
| `volume` | Sum of all values | Region-specific metric |
| `voxel_count` | Number of non-zero voxels | Proxy for region size |
| `z_filtered_mean` | Mean excluding |z| > 3 outliers | Robust to outliers |
| `z_filtered_std` | Std excluding outliers | Robust variance estimate |
| `iqr_filtered_mean` | Mean excluding IQR outliers | Alternative robust mean |
| `robust_mean` | MAD-based filtered mean | Highly robust |
| `skewness` | Asymmetry of distribution | Higher moments |
| `kurtosis` | Tail heaviness | Outlier sensitivity |

## Python API

Use the core parcellation engine directly:

```python
from parcellate import VolumetricParcellator
import nibabel as nib
import pandas as pd

# Load atlas
atlas_img = nib.load("atlas.nii.gz")
lut = pd.read_csv("atlas.tsv", sep="\t")

# Initialize parcellator
parcellator = VolumetricParcellator(
    atlas_img=atlas_img,
    lut=lut,
    background_label=0,
    resampling_target="data"  # Resample to scalar map space
)

# Parcellate a scalar map
scalar_img = nib.load("gm_map.nii.gz")
parcellator.fit(scalar_img)
stats = parcellator.transform(scalar_img)

print(stats.head())
```

### Custom Statistics

Define custom aggregation functions:

```python
def range_iqr(values):
    """Interquartile range."""
    q75, q25 = np.percentile(values[~np.isnan(values)], [75, 25])
    return q75 - q25

parcellator = VolumetricParcellator(
    atlas_img=atlas_img,
    stat_functions={"range_iqr": range_iqr}
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
