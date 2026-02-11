"""
A battery of volumetric parcellation statistics.
"""

import nibabel as nib
import numpy as np
import scipy.stats as stats

from parcellate.metrics.base import Statistic


def volume(parcel_values: np.ndarray, scalar_img: nib.Nifti1Image) -> float:
    """Compute the actual tissue volume within a mask using modulated images.

    Parameters
    ----------
    parcel_values : np.ndarray
        The modulated tissue segment values within the ROI.
    scalar_img : nib.Nifti1Image
        The modulated tissue segment (e.g., mwp1.nii).
    """
    # Calculate the volume of a single voxel in mm^3
    voxel_sizes = scalar_img.header.get_zooms()[:3]
    voxel_volume = np.prod(voxel_sizes)

    # Correct step: Sum the intensities within the mask
    # This represents the sum of tissue fractions/volume units
    tissue_sum = np.nansum(parcel_values)

    # Total volume in mm^3
    total_volume = tissue_sum * voxel_volume

    return float(total_volume)


def voxel_count(parcel_values: np.ndarray) -> int:
    """Compute the count of non-zero voxels in a parcel.

    This function counts the number of voxels in a parcel that have
    non-zero and non-NaN values. This is useful for modulated tissue
    maps (e.g., CAT12's mwp* files) where zero values indicate no
    tissue present.

    Parameters
    ----------
    parcel_values : np.ndarray
        An array of scalar values for voxels within the parcel.

    Returns
    -------
    int
        The number of non-zero, non-NaN voxels in the parcel.
    """
    num_voxels = np.sum(parcel_values.astype(bool))
    return int(num_voxels)


def _z_score_filter(values: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    """Filter values based on z-score threshold.

    Parameters
    ----------
    values : np.ndarray
        The array of values to filter.
    z_thresh : float, optional
        The z-score threshold for filtering, by default 3.0.

    Returns
    -------
    np.ndarray
        The filtered values, or original values if std is 0.
    """
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    if std_val == 0:
        return values
    z_scores = (values - mean_val) / std_val
    return values[np.abs(z_scores) <= z_thresh]


def _iqr_filter(values: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Filter values based on IQR bounds.

    Parameters
    ----------
    values : np.ndarray
        The array of values to filter.
    factor : float, optional
        The IQR factor for filtering, by default 1.5.

    Returns
    -------
    np.ndarray
        The filtered values.
    """
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return values[(values >= lower_bound) & (values <= upper_bound)]


def _robust_filter(values: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Filter values using MAD-based modified z-scores.

    Parameters
    ----------
    values : np.ndarray
        The array of values to filter.
    threshold : float, optional
        The modified z-score threshold for filtering, by default 3.5.

    Returns
    -------
    np.ndarray
        The filtered values, or original values if MAD is 0.
    """
    median_val = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median_val))
    if mad == 0:
        return values
    modified_z_scores = 0.6745 * (values - median_val) / mad
    return values[np.abs(modified_z_scores) <= threshold]


def z_filtered_mean(values: np.ndarray, z_thresh: float = 3.0) -> float:
    """Compute the mean of values after applying a z-score filter.

    Parameters
    ----------
    values : np.ndarray
        The array of values to filter and compute the mean from.
    z_thresh : float, optional
        The z-score threshold for filtering, by default 3.0.

    Returns
    -------
    float
        The mean of the filtered values.
    """
    filtered = _z_score_filter(values, z_thresh)
    if filtered.size == 0:
        return float(np.nanmean(values))
    return float(np.nanmean(filtered))


def z_filtered_std(values: np.ndarray, z_thresh: float = 3.0) -> float:
    """Compute the standard deviation of values after applying a z-score filter.

    Parameters
    ----------
    values : np.ndarray
        The array of values to filter and compute the standard deviation from.
    z_thresh : float, optional
        The z-score threshold for filtering, by default 3.0.

    Returns
    -------
    float
        The standard deviation of the filtered values.
    """
    # If std is 0, all values are identical, so filtered std is also 0
    if np.nanstd(values) == 0:
        return 0.0
    filtered = _z_score_filter(values, z_thresh)
    if filtered.size == 0:
        return float(np.nanstd(values))
    return float(np.nanstd(filtered))


def iqr_filtered_mean(values: np.ndarray, factor: float = 1.5) -> float:
    """Compute the mean of values after applying an interquartile range (IQR) filter.

    Parameters
    ----------
    values : np.ndarray
        The array of values to filter and compute the mean from.
    factor : float, optional
        The IQR factor for filtering, by default 1.5.

    Returns
    -------
    float
        The mean of the filtered values.
    """
    filtered = _iqr_filter(values, factor)
    if filtered.size == 0:
        return float(np.nanmean(values))
    return float(np.nanmean(filtered))


def iqr_filtered_std(values: np.ndarray, factor: float = 1.5) -> float:
    """Compute the standard deviation of values after applying an interquartile range (IQR) filter.

    Parameters
    ----------
    values : np.ndarray
        The array of values to filter and compute the standard deviation from.
    factor : float, optional
        The IQR factor for filtering, by default 1.5.

    Returns
    -------
    float
        The standard deviation of the filtered values.
    """
    filtered = _iqr_filter(values, factor)
    if filtered.size == 0:
        return float(np.nanstd(values))
    return float(np.nanstd(filtered))


def robust_mean(values: np.ndarray) -> float:
    """Compute the robust mean of values using median and MAD.

    Parameters
    ----------
    values : np.ndarray
        The array of values to compute the robust mean from.

    Returns
    -------
    float
        The robust mean of the values.
    """
    filtered = _robust_filter(values)
    if filtered.size == 0:
        return float(np.nanmedian(values))
    return float(np.nanmean(filtered))


def robust_std(values: np.ndarray) -> float:
    """Compute the robust standard deviation of values using median and MAD.

    Parameters
    ----------
    values : np.ndarray
        The array of values to compute the robust standard deviation from.

    Returns
    -------
    float
        The robust standard deviation of the values.
    """
    filtered = _robust_filter(values)
    if filtered.size == 0:
        return float(np.nanstd(values))
    return float(np.nanstd(filtered))


def mad_median(values: np.ndarray) -> float:
    """Compute the median absolute deviation (MAD) of values.

    Parameters
    ----------
    values : np.ndarray
        The array of values to compute the MAD from.

    Returns
    -------
    float
        The MAD of the values.
    """
    median_val = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median_val))
    return float(mad)


def volsum(values: np.ndarray) -> float:
    """Compute the sum of values.

    Parameters
    ----------
    values : np.ndarray
        The array of values to compute the sum from.

    Returns
    -------
    float
        The sum of the values.
    """
    return float(np.nansum(values))


def skewness(values: np.ndarray) -> float:
    """Compute the skewness of values.

    Parameters
    ----------
    values : np.ndarray
        The array of values to compute the skewness from.

    Returns
    -------
    float
        The skewness of the values.
    """
    return float(stats.skew(values, nan_policy="omit"))


def kurtosis(values: np.ndarray) -> float:
    """Compute the kurtosis of values.

    Parameters
    ----------
    values : np.ndarray
        The array of values to compute the kurtosis from.

    Returns
    -------
    float
        The kurtosis of the values.
    """
    return float(stats.kurtosis(values, nan_policy="omit"))


# define builtin statistics
BUILTIN_STATISTICS: list[Statistic] = [
    Statistic(name="volume_mm3", function=volume, requires_image=True),
    Statistic(name="voxel_count", function=voxel_count),
    Statistic(name="z_filtered_mean", function=z_filtered_mean),
    Statistic(name="z_filtered_std", function=z_filtered_std),
    Statistic(name="iqr_filtered_mean", function=iqr_filtered_mean),
    Statistic(name="iqr_filtered_std", function=iqr_filtered_std),
    Statistic(name="robust_mean", function=robust_mean),
    Statistic(name="robust_std", function=robust_std),
    Statistic(name="mad_median", function=mad_median),
    Statistic(name="mean", function=np.nanmean),
    Statistic(name="std", function=np.nanstd),
    Statistic(name="median", function=np.nanmedian),
    Statistic(name="sum", function=volsum),
    Statistic(name="skewness", function=skewness),
    Statistic(name="kurtosis", function=kurtosis),
]
