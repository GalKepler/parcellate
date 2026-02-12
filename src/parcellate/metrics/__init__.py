"""Statistical metrics for summarizing parcellated regions.

Provides built-in statistics (mean, median, robust estimates, higher-order moments)
and utilities for defining custom aggregation functions.
"""

from parcellate.metrics.base import Statistic
from parcellate.metrics.volume import (
    BUILTIN_STATISTICS,
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
