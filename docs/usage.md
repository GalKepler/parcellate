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

## Probabilistic (4D) atlases

Some atlases encode each region as a continuous probability map in a separate 3D volume, rather than assigning a single integer label per voxel. Examples include XTRACT white-matter tracts and the Harvard-Oxford cortical atlas. ``parcellate`` handles these natively — just pass the 4D NIfTI and the API stays the same.

**How it works**

Each volume in the 4D image corresponds to one region. Volume index 0 (zero-based) maps to region index 1 (one-based) in the output table, volume 1 maps to region 2, and so on. Within each volume, voxels whose probability is strictly greater than ``atlas_threshold`` are included in that region's statistics, so voxels can belong to more than one region simultaneously.

**Parameters**

- ``atlas_threshold`` (float, default ``0.0``) — minimum probability to include a voxel. The comparison is strict (``>``), so a voxel with probability exactly equal to the threshold is excluded. Set to ``0.0`` to include every non-zero voxel.

**Output filename entity**

When ``atlas_threshold > 0``, the value is embedded in the output filename as the BIDS-style entity ``atlasthr-<value>`` (e.g., ``atlasthr-0.25``). The entity is omitted when the threshold is zero.

```python
from parcellate import VolumetricParcellator

# 4D probabilistic atlas — detected automatically
parcellator = VolumetricParcellator(
    atlas_img="xtract_tracts.nii.gz",
    lut="xtract_lut.tsv",
    atlas_threshold=0.25,  # keep only high-confidence voxels
)
parcellator.fit("subject_FA.nii.gz")
regional_stats = parcellator.transform("subject_FA.nii.gz")
# regional_stats contains one row per tract (region)
```

## Mask thresholding

When supplying a probabilistic brain mask (e.g., a grey-matter partial-volume estimate), you may want to restrict the analysis to voxels with a high tissue probability rather than any non-zero value.

**Parameters**

- ``mask_threshold`` (float, default ``0.0``) — minimum mask value to keep a voxel. Strict ``>`` comparison; voxels equal to the threshold are excluded. The default of ``0.0`` preserves the pre-v0.2.0 behaviour of including all non-zero mask voxels.

**Output filename entity**

When ``mask_threshold > 0``, the value appears as ``maskthr-<value>`` in the output filename. The entity is omitted when the threshold is zero.

**Environment variable**

When using the ``parcellate-cat12`` standalone CLI, set ``MASKING_THRESHOLD`` to supply the threshold without editing config files.

```python
from parcellate import VolumetricParcellator

parcellator = VolumetricParcellator(
    atlas_img="atlas.nii.gz",
    mask="gm",           # MNI152 grey-matter probability map
    mask_threshold=0.5,  # include only voxels where GM probability > 0.5
)
parcellator.fit("subject_T1w.nii.gz")
regional_stats = parcellator.transform("subject_T1w.nii.gz")
```
