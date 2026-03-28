# Getting started

This guide walks through installing `parcellate` and running your first parcellation.

## Installation

```bash
pip install parcellate
```

Or install from a local checkout:

```bash
git clone https://github.com/GalKepler/parcellate.git
cd parcellate
pip install -e .
```

## Verifying your environment

`parcellate` depends on **Nibabel**, **NumPy**, and **pandas**. To confirm your installation:

```python
>>> from parcellate import VolumetricParcellator
```

If that import succeeds, you're ready to go.

---

## Quick start

The snippet below demonstrates the essential steps: load an atlas, connect a lookup table, and compute parcel-wise statistics.

```python
import nibabel as nib
import pandas as pd
from parcellate import VolumetricParcellator

# Create the parcellator
parcellator = VolumetricParcellator(
    atlas_img="atlas.nii.gz",
    lut="atlas_lut.tsv",  # TSV with "index" and "label" columns
)

# Fit and transform a scalar map
parcellator.fit("subject_T1w.nii.gz")
regional_stats = parcellator.transform("subject_T1w.nii.gz")
print(regional_stats.head())
```

The output is a `pandas.DataFrame` with one row per atlas region and one column per statistic. By default all 45 built-in statistics are computed. Use `stat_tier="core"` or `stat_tier="extended"` to compute fewer columns.

---

## Next steps

| Guide | Description |
|-------|-------------|
| [Usage guide](usage.md) | Atlases, masks, resampling, probabilistic atlases, custom statistics |
| [Metrics reference](metrics_reference.md) | All 45 statistics organized by tier |
| [API reference](api.md) | Full API documentation |
