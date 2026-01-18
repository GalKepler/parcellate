"""
A battery of volumetric parcellation statistics.
"""

import nibabel as nib
import numpy as np

from parcellate.metrics.base import Statistic


def volume(parcel_mask: np.ndarray, scalar_img: nib.Nifti1Image) -> float:
    """Compute the actual tissue volume within a mask using modulated images.

    Parameters
    ----------
    parcel_mask : np.ndarray
        A boolean mask (or integer labels) of the ROI.
    scalar_img : nib.Nifti1Image
        The modulated tissue segment (e.g., mwp1.nii).
    """
    # Load the intensity data from the image
    scalar_data = scalar_img.get_fdata()

    # Ensure the mask is boolean
    mask = parcel_mask.astype(bool)

    # Calculate the volume of a single voxel in mm^3
    voxel_sizes = scalar_img.header.get_zooms()[:3]
    voxel_volume = np.prod(voxel_sizes)

    # Correct step: Sum the intensities within the mask
    # This represents the sum of tissue fractions/volume units
    tissue_sum = np.sum(scalar_data[mask])

    # Total volume in mm^3
    total_volume = tissue_sum * voxel_volume

    return float(total_volume)


def voxel_count(parcel_mask: np.ndarray) -> int:
    """Compute the voxel count of a parcel.

    Parameters
    ----------
    scalar_img : nib.Nifti1Image
        The scalar image from which to compute voxel count.
    parcel_mask : np.ndarray
        A boolean mask of the parcel within the scalar image.

    Returns
    -------
    int
        The number of voxels in the parcel.
    """
    num_voxels = np.sum(parcel_mask.astype(bool))
    return int(num_voxels)


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
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    if std_val == 0:
        return float(mean_val)

    z_scores = (values - mean_val) / std_val
    filtered_values = values[np.abs(z_scores) <= z_thresh]
    if filtered_values.size == 0:
        return float(mean_val)

    return float(np.nanmean(filtered_values))


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
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    if std_val == 0:
        return float(mean_val)

    z_scores = (values - mean_val) / std_val
    filtered_values = values[np.abs(z_scores) <= z_thresh]
    if filtered_values.size == 0:
        return float(std_val)

    return float(np.nanstd(filtered_values))


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
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
    if filtered_values.size == 0:
        return float(np.nanmean(values))

    return float(np.nanmean(filtered_values))


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
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
    if filtered_values.size == 0:
        return float(np.nanstd(values))

    return float(np.nanstd(filtered_values))


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
    median_val = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median_val))
    if mad == 0:
        return float(median_val)

    modified_z_scores = 0.6745 * (values - median_val) / mad
    filtered_values = values[np.abs(modified_z_scores) <= 3.5]
    if filtered_values.size == 0:
        return float(median_val)

    return float(np.nanmean(filtered_values))


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
    median_val = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median_val))
    if mad == 0:
        return float(np.nanstd(values))

    modified_z_scores = 0.6745 * (values - median_val) / mad
    filtered_values = values[np.abs(modified_z_scores) <= 3.5]
    if filtered_values.size == 0:
        return float(np.nanstd(values))

    return float(np.nanstd(filtered_values))


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
]
