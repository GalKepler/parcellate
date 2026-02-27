# Getting started

This guide walks through installing ``parcellate`` and running your first parcellation.

## Installation

```bash
pip install parcellate
```

For optional `.env` file support in the legacy `parcellate-cat12` CLI:

```bash
pip install parcellate[dotenv]
```

Alternatively, install from a local checkout:

```bash
git clone https://github.com/GalKepler/parcellate.git
cd parcellate
pip install -e .
```

## Verifying your environment

``parcellate`` depends on scientific Python libraries such as **Nibabel**, **NumPy**, and **pandas**. To confirm your installation, open a Python shell and import the core components:

```python
>>> import nibabel as nib
>>> import numpy as np
>>> import pandas as pd
>>> from parcellate import VolumetricParcellator
```

If those imports succeed, you can move on to the usage examples below.

---

## Quick start: BIDS App (CLI)

`parcellate` follows the [BIDS App](https://bids-apps.neuroimaging.io/) convention. The basic invocation is:

```
parcellate <bids_dir> <output_dir> participant --pipeline {cat12,qsirecon} --config config.toml
```

### CAT12 example

Create a minimal TOML configuration file:

```toml
# atlases.toml
[[atlases]]
name  = "schaefer400"
path  = "/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
lut   = "/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152.tsv"
space = "MNI152NLin2009cAsym"
```

Then run:

```bash
parcellate /data/cat12_derivatives /data/parcellations participant \
    --pipeline cat12 \
    --config atlases.toml
```

For a detailed guide to CAT12 inputs and outputs, see [CAT12 pipeline guide](cat12_guide.md).

### QSIRecon example

```toml
# atlases.toml
[[atlases]]
name  = "jhu"
path  = "/atlases/JHU-ICBM-labels-1mm.nii.gz"
lut   = "/atlases/JHU-ICBM-labels-1mm.tsv"
space = "MNI152NLin6Asym"
```

```bash
parcellate /data/qsirecon_derivatives /data/parcellations participant \
    --pipeline qsirecon \
    --config atlases.toml \
    --stat-tier core
```

For a detailed guide to QSIRecon inputs and outputs, see [QSIRecon pipeline guide](qsirecon_guide.md).

---

## Quick start: Python API

The snippet below demonstrates the essential steps: load an atlas, connect a lookup table, and compute parcel-wise statistics.

```python
import nibabel as nib
import pandas as pd
from parcellate import VolumetricParcellator

# Load a labeled atlas and its lookup table
atlas = nib.load("atlas.nii.gz")
lut = pd.read_csv("atlas_lut.tsv", sep="\t")

# Create the parcellator
parcellator = VolumetricParcellator(atlas_img=atlas, lut=lut)

# Fit and evaluate a scalar map
parcellator.fit("subject_T1w.nii.gz")
regional_stats = parcellator.transform("subject_T1w.nii.gz")
print(regional_stats.head())
```

The output is a `pandas.DataFrame` with one row per atlas region and one column per statistic. By default all 45 built-in statistics are computed. Use `stat_tier="core"` or `stat_tier="extended"` to compute fewer columns — see [Metrics reference](metrics_reference.md).

---

## Next steps

| Guide | Description |
|-------|-------------|
| [CAT12 pipeline guide](cat12_guide.md) | Input layout, output format, masking, TIV extraction |
| [QSIRecon pipeline guide](qsirecon_guide.md) | Input layout, output format, probabilistic atlases |
| [CLI reference](cli_reference.md) | All CLI flags and options |
| [Metrics reference](metrics_reference.md) | All 45 statistics organized by tier |
| [Configuration reference](configuration.md) | TOML configuration file format |
| [Troubleshooting](troubleshooting.md) | Common errors and solutions |
