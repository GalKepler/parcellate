# Metrics reference

`parcellate` ships with 45 built-in statistics organised into named **tiers**.
Select a tier via `stat_tier` in Python, TOML config, or the `--stat-tier` CLI flag.

## Tiers at a glance

| Tier | Count | Use case |
|------|-------|----------|
| `core` | 6 | Fast exploration, large cohorts |
| `extended` | 21 | Production pipelines |
| `diagnostic` / `all` | 45 | Data quality, distribution inspection |

Default: `diagnostic` (all 45 statistics).

---

## Core tier (6 statistics)

The six most common descriptors. Suitable when output file size or compute
time matters most.

| Name | Description |
|------|-------------|
| `mean` | Arithmetic mean (`np.nanmean`). |
| `std` | Standard deviation (`np.nanstd`). |
| `median` | Median (`np.nanmedian`). |
| `volume_mm3` | Region volume in mm³ (uses voxel dimensions from the image header). |
| `voxel_count` | Number of non-background voxels in the region. |
| `sum` | Sum of voxel intensities. |

---

## Extended tier (21 statistics)

Everything in *core* plus robust/filtered estimates and basic shape descriptors.
A good default for production parcellation pipelines.

### Robust / filtered means and dispersions

| Name | Description |
|------|-------------|
| `robust_mean` | Mean after removing voxels more than 3 median absolute deviations (MAD) from the median. |
| `robust_std` | Standard deviation of the MAD-filtered sample. |
| `mad_median` | Median absolute deviation from the median. |
| `z_filtered_mean` | Mean after discarding voxels with `|z| > 3`. |
| `z_filtered_std` | Standard deviation of the z-score-filtered sample. |
| `iqr_filtered_mean` | Mean after removing voxels outside the interquartile fence. |
| `iqr_filtered_std` | Standard deviation of the IQR-filtered sample. |
| `cv` | Coefficient of variation: `std / mean`. |
| `robust_cv` | Robust coefficient of variation: `IQR / median`. |

### Shape

| Name | Description |
|------|-------------|
| `skewness` | Third standardised moment. Positive = right-skewed. |
| `excess_kurtosis` | Fourth standardised moment minus 3. Positive = heavier tails than normal. |

### Key percentiles

| Name | Description |
|------|-------------|
| `percentile_5` | 5th percentile. |
| `percentile_25` | 25th percentile (Q1). |
| `percentile_75` | 75th percentile (Q3). |
| `percentile_95` | 95th percentile. |

---

## Diagnostic tier — additional statistics (beyond extended)

### Additional shape metrics

| Name | Description |
|------|-------------|
| `quartile_dispersion` | `(Q3 − Q1) / (Q3 + Q1)` — scale-free dispersion. |
| `abs_skewness` | Absolute value of skewness. |
| `abs_excess_kurtosis` | Absolute value of excess kurtosis. |
| `bimodality_coefficient` | Bimodality coefficient `(skewness² + 1) / kurtosis`. Values > 0.555 suggest bimodality. |

### Outlier proportions

| Name | Description |
|------|-------------|
| `prop_outliers_2sd` | Proportion of voxels more than 2 SD from the mean. |
| `prop_outliers_3sd` | Proportion of voxels more than 3 SD from the mean. |
| `prop_outliers_iqr` | Proportion of voxels outside 1.5 × IQR fence (Tukey fences). |

### Tail behaviour

| Name | Description |
|------|-------------|
| `left_tail_mass` | Proportion of values below `mean − 2σ`. |
| `right_tail_mass` | Proportion of values above `mean + 2σ`. |
| `tail_asymmetry` | `right_tail_mass − left_tail_mass`. Positive = heavier right tail. |
| `excess_tail_mass` | `left_tail_mass + right_tail_mass`. Total extreme-value proportion. |

### Normality tests

| Name | Description |
|------|-------------|
| `dagostino_k2` | D'Agostino K² test statistic (combines skewness and kurtosis). |
| `dagostino_p` | P-value for the K² test. Small values indicate non-normality. |
| `log_dagostino_k2` | Natural log of `dagostino_k2` (compresses dynamic range). |
| `shapiro_w` | Shapiro-Wilk W statistic (only for 3 ≤ n ≤ 5000; `NaN` otherwise). |
| `shapiro_p` | Shapiro-Wilk p-value. |
| `qq_correlation` | Pearson correlation of sorted values against theoretical normal quantiles. |
| `qq_r_squared` | R² of the Q-Q linear fit. Values near 1 indicate normality. |

### Information theory

| Name | Description |
|------|-------------|
| `histogram_entropy` | Shannon entropy of a normalised histogram (20 bins). |

### Additional percentiles

| Name | Description |
|------|-------------|
| `percentile_10` | 10th percentile. |
| `percentile_50` | 50th percentile (median, same as `median`). |
| `percentile_90` | 90th percentile. |

### Boolean flags

These columns contain `True`/`False` values and are useful for quick QC filtering.

| Name | Threshold |
|------|-----------|
| `is_strongly_skewed` | `|skewness| > 1` |
| `is_heavy_tailed` | `excess_kurtosis > 1` |
| `is_bimodal` | `bimodality_coefficient > 0.555` |
| `has_outliers` | `prop_outliers_iqr > 0.05` |
| `fails_normality` | `dagostino_p < 0.05` |

---

## Using tiers in Python

```python
from parcellate import VolumetricParcellator

# Use the 'extended' tier
parcellator = VolumetricParcellator(
    atlas_img="atlas.nii.gz",
    lut="atlas_lut.tsv",
    stat_tier="extended",
)
parcellator.fit("subject_T1w.nii.gz")
df = parcellator.transform("subject_T1w.nii.gz")
# df has columns: index, label, mean, std, median, volume_mm3, ...
```

## Custom statistics

You can replace the built-in statistics entirely with your own functions:

```python
import numpy as np
from parcellate import VolumetricParcellator

parcellator = VolumetricParcellator(
    atlas_img="atlas.nii.gz",
    stat_functions={
        "trimmed_mean": lambda x: float(np.nanmean(x[np.abs(x - np.nanmedian(x)) < 2 * np.nanstd(x)])),
        "q90": lambda x: float(np.nanpercentile(x, 90)),
    },
)
```

When `stat_functions` is provided it takes precedence over `stat_tier`.

## Accessing tier lists programmatically

```python
from parcellate.metrics import CORE_STATISTICS, EXTENDED_STATISTICS, STATISTIC_TIERS

# Inspect available tiers
for tier_name, stats in STATISTIC_TIERS.items():
    print(f"{tier_name}: {[s.name for s in stats]}")
```
