"""Statistical metrics for summarizing parcellated regions.

Provides built-in statistics (mean, median, robust estimates, higher-order moments)
and utilities for defining custom aggregation functions.

Tier system
-----------
Statistics are grouped into named tiers for convenience:

- ``"core"`` — six fundamental descriptors (mean, std, median, volume, voxel count, sum).
- ``"extended"`` — core + robust/filtered estimates and basic shape descriptors.
- ``"diagnostic"`` / ``"all"`` — all 45 built-in statistics (default).

Pass a tier name to :class:`~parcellate.parcellation.volume.VolumetricParcellator`
via ``stat_tier="extended"`` to compute only that subset.
"""

from parcellate.metrics.base import Statistic
from parcellate.metrics.volume import (
    BUILTIN_STATISTICS,
    CORE_STATISTICS,
    DIAGNOSTIC_STATISTICS,
    EXTENDED_STATISTICS,
    STATISTIC_TIERS,
    iqr_filtered_mean,
    iqr_filtered_std,
    mad_median,
    robust_mean,
    robust_std,
    volume,
    voxel_count,
    z_filtered_mean,
    z_filtered_std,
)

__all__ = [
    "BUILTIN_STATISTICS",
    "CORE_STATISTICS",
    "DIAGNOSTIC_STATISTICS",
    "EXTENDED_STATISTICS",
    "STATISTIC_TIERS",
    "Statistic",
    "iqr_filtered_mean",
    "iqr_filtered_std",
    "mad_median",
    "robust_mean",
    "robust_std",
    "volume",
    "voxel_count",
    "z_filtered_mean",
    "z_filtered_std",
]
