# Getting started

This guide walks through installing ``parcellate`` and validating that the core workflow runs on your machine.

## Installation

```bash
pip install parcellate
```

Alternatively, install from a local checkout to ensure the package and its dependencies are available to Sphinx and Read the Docs:

```bash
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

## Minimal example

The snippet below demonstrates the essential steps: load an atlas, connect a lookup table, and compute a handful of parcel-wise statistics.

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
