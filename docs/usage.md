# Usage guide

The :class:`parcellate.parcellation.volume.VolumetricParcellator` orchestrates atlas handling, resampling, and statistic computation. This page outlines the most common workflows.

## Working with atlases and lookup tables

You can provide atlas metadata in multiple ways:

- **Lookup table**: pass a TSV file or :class:`pandas.DataFrame` with ``index`` and ``label`` columns. Missing columns raise a :class:`parcellate.parcellation.volume.MissingLUTColumnsError`.
- **Custom label selection**: supply a list or mapping of label IDs via ``labels`` to restrict the analysis to specific parcels.
- **Built-in masks**: set ``mask="gm"``, ``"wm"``, or ``"brain"`` to leverage MNI152 tissue masks from :mod:`nilearn`. Custom mask images are also supported.

```python
from parcellate import VolumetricParcellator

parcellator = VolumetricParcellator(
    atlas_img="atlas.nii.gz",
    lut="atlas_lut.tsv",
    mask="gm",  # use MNI152 grey-matter mask
    resampling_target="data",
)
```

## Running a parcellation

1. **Fit** the parcellator to set up the atlas grid and resampling strategy.
2. **Transform** scalar images to compute parcel-wise statistics.

```python
parcellator.fit("subject_T1w.nii.gz")
regional_stats = parcellator.transform("subject_T1w.nii.gz")

print(regional_stats.columns)
# index, label, volume_mm3, voxel_count, mean, std, ...
```

If ``transform`` is called before ``fit``, a :class:`parcellate.parcellation.volume.ParcellatorNotFittedError` is raised to prevent accidental misuse.

## Customizing statistics

``parcellate`` ships with robust defaults defined in :mod:`parcellate.metrics.volume`, including volume (``mm³``), voxel count, trimmed means, and robust dispersion estimators. Supply your own mapping of statistic names to callables to extend or replace the defaults:

```python
import numpy as np

custom_stats = {
    "nanmedian": np.nanmedian,
    "z_filtered_mean": lambda values: float(np.nanmean(values[np.abs(values) < 3])),
}

parcellator = VolumetricParcellator(
    atlas_img="atlas.nii.gz",
    stat_functions=custom_stats,
)
```

Each statistic receives the parcel's voxel values. To access the scalar image (for example, to compute voxel volume), set ``requires_image=True`` on a :class:`parcellate.metrics.base.Statistic` instance.

## Resampling behavior

Use ``resampling_target`` to control how atlases and scalar maps are aligned:

- ``"data"`` (default) resamples the atlas to the scalar image grid using nearest-neighbor interpolation.
- ``"labels"`` resamples scalar maps to the atlas grid, preserving atlas topology at the cost of interpolating intensities.
- ``None`` keeps both images on their native grids; set this only when inputs already align.

The helper methods :meth:`parcellate.parcellation.volume.VolumetricParcellator._prepare_map` and :meth:`parcellate.parcellation.volume.VolumetricParcellator._apply_mask_to_atlas` encapsulate the resampling steps and mask application, ensuring consistent background handling via ``background_label``.
