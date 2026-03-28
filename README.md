# parcellate

[![Build status](https://img.shields.io/github/actions/workflow/status/GalKepler/parcellate/main.yml?branch=main)](https://github.com/GalKepler/parcellate/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/GalKepler/parcellate/branch/main/graph/badge.svg)](https://codecov.io/gh/GalKepler/parcellate)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://GalKepler.github.io/parcellate/)
[![License](https://img.shields.io/github/license/GalKepler/parcellate)](https://img.shields.io/github/license/GalKepler/parcellate)

> Extract regional statistics from scalar neuroimaging maps using atlas-based parcellation.

## What It Does

**parcellate** is a Python library that extracts regional statistics from volumetric brain images using atlas-based parcellation. Given a scalar map (e.g., gray matter density, fractional anisotropy) and a labeled atlas, it computes summary statistics for each brain region and returns a tidy `pandas.DataFrame`.

**Key Features:**

- **45 built-in statistics** in three tiers (core / extended / diagnostic), including robust estimates, higher-order moments, and normality tests
- **Flexible atlases**: discrete 3D integer-label atlases and 4D probabilistic atlases (e.g., XTRACT tracts)
- **Built-in masks**: MNI152 grey-matter, white-matter, and brain masks via nilearn
- **Custom statistics**: supply any callable as a statistic
- **Scikit-learn–style API**: `fit` / `transform` pattern with smart resampling caching

## Installation

```bash
pip install parcellate
```

Or from a local checkout:

```bash
git clone https://github.com/GalKepler/parcellate.git
cd parcellate
pip install -e .
```

## Quick Start

```python
from parcellate import VolumetricParcellator

parcellator = VolumetricParcellator(
    atlas_img="atlas.nii.gz",
    lut="atlas_lut.tsv",       # TSV with "index" and "label" columns
    mask="gm",                 # built-in MNI152 grey-matter mask
    stat_tier="extended",      # 21 statistics (default: all 45)
)

parcellator.fit("subject_map.nii.gz")
stats = parcellator.transform("subject_map.nii.gz")
print(stats.head())
```

## Statistics Tiers

Select a tier with `stat_tier` to control the number of computed statistics:

| Tier | Count | Use case |
|------|-------|----------|
| `core` | 6 | Fast exploration, large cohorts |
| `extended` | 21 | Production pipelines |
| `diagnostic` (default) | 45 | QC, distribution inspection |

**Core:** `mean`, `std`, `median`, `volume_mm3`, `voxel_count`, `sum`

**Extended adds:** robust means (MAD, z-score, IQR filtered), dispersion (`cv`, `robust_cv`), shape (`skewness`, `excess_kurtosis`), percentiles (5th, 25th, 75th, 95th)

**Diagnostic adds:** normality tests (Shapiro-Wilk, D'Agostino K²), outlier proportions, tail mass, entropy, boolean QC flags

See the full [Metrics reference](https://GalKepler.github.io/parcellate/metrics_reference/) for descriptions of all 45 statistics.

## Custom Statistics

```python
import numpy as np
from parcellate import VolumetricParcellator

parcellator = VolumetricParcellator(
    atlas_img="atlas.nii.gz",
    stat_functions={
        "iqr": lambda x: float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
        "q90": lambda x: float(np.nanpercentile(x, 90)),
    },
)
```

## Output

`transform()` returns a `pandas.DataFrame` with one row per atlas region:

| Column | Description |
|--------|-------------|
| `index` | Integer region index from the atlas |
| `label` | Region name from the lookup table |
| `mean`, `std`, … | Computed statistics (depends on `stat_tier`) |

## Development

```bash
make install   # create venv + install pre-commit hooks
make test      # run tests with coverage
make check     # lock file, pre-commit, deptry
```

## Documentation

Full documentation: [https://GalKepler.github.io/parcellate/](https://GalKepler.github.io/parcellate/)

## License

MIT License — see the LICENSE file for details.

## Citation

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
