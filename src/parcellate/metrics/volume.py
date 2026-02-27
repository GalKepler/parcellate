"""
A battery of volumetric parcellation statistics.
"""

import nibabel as nib
import numpy as np
from scipy.stats import entropy, iqr, normaltest, probplot, shapiro, skew
from scipy.stats import kurtosis as scipy_kurtosis

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


# =============================================================================
# COEFFICIENT OF VARIATION
# =============================================================================


def cv(values):
    """
    Classical coefficient of variation (CV = SD / mean).

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Coefficient of variation. Higher values indicate more relative variability.

    Notes
    -----
    Sensitive to mean approaching zero. For metrics that can be zero or negative,
    consider using robust_cv instead.

    Typical values for neuroimaging:
    - Cortical thickness: CV ~ 0.10
    - FA: CV ~ 0.15
    - MD: CV ~ 0.20
    """
    return np.std(values) / (np.mean(values) + 1e-10)


def robust_cv(values):
    """
    Robust coefficient of variation (IQR / median).

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Robust coefficient of variation. Less sensitive to outliers than classical CV.

    Notes
    -----
    Recommended when outliers are present or suspected. More stable than classical
    CV for distributions with heavy tails.
    """
    return iqr(values) / (np.median(values) + 1e-10)


def quartile_dispersion(values):
    """
    Quartile coefficient of dispersion.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Quartile dispersion coefficient: (Q3 - Q1) / (Q3 + Q1)

    Notes
    -----
    Scale-free measure of dispersion. Bounded between 0 and 1.
    Less affected by extreme values than CV.
    """
    clean = np.asarray(values)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    q1, q3 = np.percentile(clean, [25, 75])
    return (q3 - q1) / (q3 + q1 + 1e-10)


# =============================================================================
# SHAPE METRICS
# =============================================================================


def skewness(values):
    """
    Skewness (third standardized moment).

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Skewness value. 0 = symmetric, >0 = right tail, <0 = left tail.

    Notes
    -----
    Problematic if |skewness| > 0.5.

    Neuroimaging context:
    - Cortical thickness often shows negative skew (-0.2 to -0.8) due to
      partial volume effects
    - FA often shows positive skew (0.3 to 1.2) due to crossing fibers
    """
    return float(skew(values, nan_policy="omit"))


def excess_kurtosis(values):
    """
    Excess kurtosis (fourth standardized moment minus 3).

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Excess kurtosis. 0 = normal, >0 = heavy tails, <0 = light tails.

    Notes
    -----
    Problematic if |kurtosis| > 1.0.

    Positive kurtosis indicates outlier-prone distributions. Common in diffusion
    metrics due to partial volume effects with CSF.
    """
    return float(scipy_kurtosis(values, nan_policy="omit"))


def abs_skewness(values):
    """
    Absolute value of skewness.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Absolute skewness. Useful for categorizing without caring about direction.
    """
    return float(np.abs(skew(values, nan_policy="omit")))


def abs_excess_kurtosis(values):
    """
    Absolute value of excess kurtosis.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Absolute excess kurtosis.
    """
    return float(np.abs(scipy_kurtosis(values, nan_policy="omit")))


# =============================================================================
# BIMODALITY
# =============================================================================


def bimodality_coefficient(values):
    """
    Bimodality coefficient (Pfister et al. 2013).

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Bimodality coefficient. Values > 0.555 suggest bimodal/multimodal distribution.

    Notes
    -----
    Calculated as: BC = (skew² + 1) / (kurtosis + 3(n-1)²/((n-2)(n-3)))

    Critical for cortical thickness ROIs that span gyri and sulci, which often
    show bimodal distributions.

    References
    ----------
    Pfister R, Schwarz KA, Janczyk M, Dale R, Freeman JB (2013).
    Good things peak in pairs: a note on the bimodality coefficient.
    Front Psychol 4:700.
    """
    n = len(values)
    if n < 4:
        return np.nan

    skew_val = skew(values, nan_policy="omit")
    kurt_val = scipy_kurtosis(values, nan_policy="omit")

    m3_squared = skew_val**2
    m4 = kurt_val + 3  # Convert excess kurtosis to raw kurtosis

    bc = (m3_squared + 1) / (m4 + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3)))
    return float(bc)


# =============================================================================
# OUTLIER PREVALENCE
# =============================================================================


def prop_outliers_2sd(values):
    """
    Proportion of values beyond ±2 standard deviations from mean.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Proportion of outliers (0 to 1). Expected ~0.046 for normal distribution.

    Notes
    -----
    Problematic if > 0.10 (10%).
    """
    mean, std = np.mean(values), np.std(values)
    return np.mean(np.abs(values - mean) > 2 * std)


def prop_outliers_3sd(values):
    """
    Proportion of values beyond ±3 standard deviations from mean.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Proportion of outliers (0 to 1). Expected ~0.003 for normal distribution.

    Notes
    -----
    Problematic if > 0.02 (2%).
    High values suggest segmentation errors or image quality issues.
    """
    mean, std = np.mean(values), np.std(values)
    return np.mean(np.abs(values - mean) > 3 * std)


def prop_outliers_iqr(values):
    """
    Proportion of Tukey fence outliers (beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR).

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Proportion of outliers (0 to 1). Expected ~0.007 for normal distribution.

    Notes
    -----
    More robust than SD-based outlier detection. This is the standard boxplot
    definition of outliers. Problematic if > 0.05 (5%).
    """
    clean = np.asarray(values)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    q1, q3 = np.percentile(clean, [25, 75])
    iqr_val = q3 - q1
    lower_fence = q1 - 1.5 * iqr_val
    upper_fence = q3 + 1.5 * iqr_val
    return float(np.mean((clean < lower_fence) | (clean > upper_fence)))


# =============================================================================
# TAIL BEHAVIOR
# =============================================================================


def left_tail_mass(values):
    """
    Proportion of values in left tail (below mean - 2*SD).

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Left tail mass. Expected ~0.023 for normal distribution.
    """
    mean, std = np.mean(values), np.std(values)
    return np.mean(values < (mean - 2 * std))


def right_tail_mass(values):
    """
    Proportion of values in right tail (above mean + 2*SD).

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Right tail mass. Expected ~0.023 for normal distribution.
    """
    mean, std = np.mean(values), np.std(values)
    return np.mean(values > (mean + 2 * std))


def tail_asymmetry(values):
    """
    Difference between right and left tail masses.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Tail asymmetry. >0 = heavier right tail, <0 = heavier left tail.

    Notes
    -----
    Expected ~0 for symmetric distributions. Problematic if |asymmetry| > 0.02.
    Helps identify which tail is causing distributional problems.
    """
    return right_tail_mass(values) - left_tail_mass(values)


def excess_tail_mass(values):
    """
    Total excess tail mass beyond what's expected for normal distribution.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        Excess tail mass. >0 indicates heavy tails.

    Notes
    -----
    For normal distribution, total tail mass (beyond ±2SD) = 4.6%.
    Excess tail mass = observed - expected.
    Problematic if > 0.02 (2% excess).
    """
    normal_tail_mass = 0.023  # Each tail
    total_observed = left_tail_mass(values) + right_tail_mass(values)
    return total_observed - 2 * normal_tail_mass


# =============================================================================
# NORMALITY TESTS
# =============================================================================


def dagostino_k2(values):
    """
    D'Agostino-Pearson K² statistic for normality test.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values (requires n ≥ 20)

    Returns
    -------
    float
        K² test statistic (chi-squared distributed with 2 df)

    Notes
    -----
    Returns NaN if n < 20. Higher values indicate stronger deviation from normality.

    See also
    --------
    dagostino_p : Corresponding p-value
    """
    if len(values) < 20:
        return np.nan
    k_stat, _ = normaltest(values)
    return k_stat


def dagostino_p(values):
    """
    D'Agostino-Pearson p-value for normality test.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values (requires n ≥ 20)

    Returns
    -------
    float
        p-value. Values < 0.05 reject null hypothesis of normality.

    Notes
    -----
    Returns NaN if n < 20. Combines tests of skewness and kurtosis.
    More computationally efficient than Shapiro-Wilk for large samples.
    """
    if len(values) < 20:
        return np.nan
    _, p_val = normaltest(values)
    return p_val


def log_dagostino_k2(values):
    """
    Log-transformed D'Agostino K² statistic.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values (requires n ≥ 20)

    Returns
    -------
    float
        log(K²). Useful for visualization when K² spans many orders of magnitude.
    """
    k_stat = dagostino_k2(values)
    if np.isnan(k_stat):
        return np.nan
    return np.log(k_stat + 1e-10)


def shapiro_w(values):
    """
    Shapiro-Wilk W statistic for normality test.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values (requires 3 ≤ n ≤ 5000)

    Returns
    -------
    float
        W statistic (ranges 0 to 1). Values close to 1 indicate normality.

    Notes
    -----
    Returns NaN if n < 3 or n > 5000 or if test fails.
    Most powerful normality test but computationally expensive for large samples.
    Problematic if W < 0.95.

    See also
    --------
    shapiro_p : Corresponding p-value
    """
    if not (3 <= len(values) <= 5000):
        return np.nan
    try:
        w_stat, _ = shapiro(values)
    except Exception:
        return np.nan
    else:
        return w_stat


def shapiro_p(values):
    """
    Shapiro-Wilk p-value for normality test.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values (requires 3 ≤ n ≤ 5000)

    Returns
    -------
    float
        p-value. Values < 0.05 reject null hypothesis of normality.

    Notes
    -----
    Returns NaN if n < 3 or n > 5000 or if test fails.
    """
    if not (3 <= len(values) <= 5000):
        return np.nan
    try:
        _, p_val = shapiro(values)
    except Exception:
        return np.nan
    else:
        return p_val


# =============================================================================
# Q-Q PLOT METRICS
# =============================================================================


def qq_correlation(values):
    """
    Correlation coefficient from Q-Q plot (sample vs. theoretical normal quantiles).

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        R value from Q-Q plot (ranges -1 to 1). Values close to 1 indicate normality.

    Notes
    -----
    Quantifies the linearity of the Q-Q plot. Problematic if R² < 0.95.

    See also
    --------
    qq_r_squared : R² version (more commonly reported)
    """
    (_osm, _osr), (_slope, _intercept, r) = probplot(values, dist="norm", fit=True)
    return r


def qq_r_squared(values):
    """
    R² from Q-Q plot regression.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    float
        R² (ranges 0 to 1). Values close to 1 indicate good fit to normal distribution.

    Notes
    -----
    Problematic if R² < 0.95. Single-number summary of Q-Q plot linearity.
    """
    r = qq_correlation(values)
    return r**2


# =============================================================================
# ENTROPY
# =============================================================================


def histogram_entropy(values, bins="auto"):
    """
    Shannon entropy of histogram.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values
    bins : int or str, optional
        Number of histogram bins or binning strategy. Default 'auto'.

    Returns
    -------
    float
        Entropy in bits. Higher values indicate more uniform/spread out distributions.

    Notes
    -----
    Useful for comparing distributional complexity across ROIs.
    Not directly interpretable for normality assessment.
    """
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return np.nan

    hist, _ = np.histogram(valid_values, bins=bins, density=True)
    hist = hist / hist.sum()  # Normalize to probabilities
    hist = hist[hist > 0]  # Remove zero bins
    return float(entropy(hist, base=2))


# =============================================================================
# PERCENTILES
# =============================================================================


def percentile_5(values):
    """5th percentile."""
    return float(np.nanpercentile(values, 5))


def percentile_10(values):
    """10th percentile."""
    return float(np.nanpercentile(values, 10))


def percentile_25(values):
    """25th percentile (Q1)."""
    return float(np.nanpercentile(values, 25))


def percentile_50(values):
    """50th percentile (median)."""
    return float(np.nanpercentile(values, 50))


def percentile_75(values):
    """75th percentile (Q3)."""
    return float(np.nanpercentile(values, 75))


def percentile_90(values):
    """90th percentile."""
    return float(np.nanpercentile(values, 90))


def percentile_95(values):
    """95th percentile."""
    return float(np.nanpercentile(values, 95))


# =============================================================================
# CATEGORICAL FLAGS
# =============================================================================


def is_strongly_skewed(values, threshold=0.5):
    """
    Whether distribution is strongly skewed.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values
    threshold : float, optional
        Absolute skewness threshold. Default 0.5.

    Returns
    -------
    bool
        True if |skewness| > threshold
    """
    return abs_skewness(values) > threshold


def is_heavy_tailed(values, threshold=1.0):
    """
    Whether distribution has heavy tails.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values
    threshold : float, optional
        Absolute excess kurtosis threshold. Default 1.0.

    Returns
    -------
    bool
        True if |excess_kurtosis| > threshold
    """
    return abs_excess_kurtosis(values) > threshold


def is_bimodal(values, threshold=0.555):
    """
    Whether distribution appears bimodal.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values
    threshold : float, optional
        Bimodality coefficient threshold. Default 0.555.

    Returns
    -------
    bool
        True if bimodality_coefficient > threshold
    """
    return bimodality_coefficient(values) > threshold


def has_outliers(values, threshold=0.01):
    """
    Whether distribution has excessive outliers.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values
    threshold : float, optional
        Proportion threshold for 3SD outliers. Default 0.01 (1%).

    Returns
    -------
    bool
        True if proportion of 3SD outliers > threshold
    """
    return prop_outliers_3sd(values) > threshold


def fails_normality(values):
    """
    Whether distribution fails normality tests.

    Parameters
    ----------
    values : array-like
        Within-ROI voxel/vertex values

    Returns
    -------
    bool
        True if Shapiro-Wilk p < 0.05 (if available), else D'Agostino p < 0.05

    Notes
    -----
    Prioritizes Shapiro-Wilk (more powerful) when n ≤ 5000.
    Falls back to D'Agostino-Pearson for larger samples.
    """
    shapiro_p_val = shapiro_p(values)
    if not np.isnan(shapiro_p_val):
        return shapiro_p_val < 0.05

    dagostino_p_val = dagostino_p(values)
    if not np.isnan(dagostino_p_val):
        return dagostino_p_val < 0.05

    return False  # If no test available, default to False


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

#: Names of statistics in the *core* tier.
#: Suitable for fast exploratory analysis.  Includes only the most common
#: descriptive measures (mean, standard deviation, median, volume, voxel
#: count, and sum).
CORE_STATISTIC_NAMES: frozenset[str] = frozenset({
    "mean",
    "std",
    "median",
    "volume_mm3",
    "voxel_count",
    "sum",
})

#: Names of statistics in the *extended* tier.
#: Adds robust/filtered estimates and basic shape descriptors on top of the
#: *core* set — a good default for production parcellation pipelines.
EXTENDED_STATISTIC_NAMES: frozenset[str] = frozenset({
    "mean",
    "std",
    "median",
    "volume_mm3",
    "voxel_count",
    "sum",
    "robust_mean",
    "robust_std",
    "mad_median",
    "z_filtered_mean",
    "z_filtered_std",
    "iqr_filtered_mean",
    "iqr_filtered_std",
    "cv",
    "robust_cv",
    "skewness",
    "excess_kurtosis",
    "percentile_5",
    "percentile_25",
    "percentile_75",
    "percentile_95",
})

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
    Statistic(name="cv", function=cv),
    Statistic(name="robust_cv", function=robust_cv),
    Statistic(name="quartile_dispersion", function=quartile_dispersion),
    Statistic(name="skewness", function=skewness),
    Statistic(name="excess_kurtosis", function=excess_kurtosis),
    Statistic(name="abs_skewness", function=abs_skewness),
    Statistic(name="abs_excess_kurtosis", function=abs_excess_kurtosis),
    Statistic(name="bimodality_coefficient", function=bimodality_coefficient),
    Statistic(name="prop_outliers_2sd", function=prop_outliers_2sd),
    Statistic(name="prop_outliers_3sd", function=prop_outliers_3sd),
    Statistic(name="prop_outliers_iqr", function=prop_outliers_iqr),
    Statistic(name="left_tail_mass", function=left_tail_mass),
    Statistic(name="right_tail_mass", function=right_tail_mass),
    Statistic(name="tail_asymmetry", function=tail_asymmetry),
    Statistic(name="excess_tail_mass", function=excess_tail_mass),
    Statistic(name="dagostino_k2", function=dagostino_k2),
    Statistic(name="dagostino_p", function=dagostino_p),
    Statistic(name="log_dagostino_k2", function=log_dagostino_k2),
    Statistic(name="shapiro_w", function=shapiro_w),
    Statistic(name="shapiro_p", function=shapiro_p),
    Statistic(name="qq_correlation", function=qq_correlation),
    Statistic(name="qq_r_squared", function=qq_r_squared),
    Statistic(name="histogram_entropy", function=histogram_entropy),
    Statistic(name="percentile_5", function=percentile_5),
    Statistic(name="percentile_10", function=percentile_10),
    Statistic(name="percentile_25", function=percentile_25),
    Statistic(name="percentile_50", function=percentile_50),
    Statistic(name="percentile_75", function=percentile_75),
    Statistic(name="percentile_90", function=percentile_90),
    Statistic(name="percentile_95", function=percentile_95),
    Statistic(name="is_strongly_skewed", function=is_strongly_skewed),
    Statistic(name="is_heavy_tailed", function=is_heavy_tailed),
    Statistic(name="is_bimodal", function=is_bimodal),
    Statistic(name="has_outliers", function=has_outliers),
    Statistic(name="fails_normality", function=fails_normality),
]

# ---------------------------------------------------------------------------
# Tier lists — built from BUILTIN_STATISTICS so ordering is stable
# ---------------------------------------------------------------------------

#: *Core* statistics: six fundamental descriptive measures.
#: Suitable for fast exploratory runs or when output file size matters.
CORE_STATISTICS: list[Statistic] = [s for s in BUILTIN_STATISTICS if s.name in CORE_STATISTIC_NAMES]

#: *Extended* statistics: core + robust estimates and basic shape descriptors.
#: A good default for production parcellation pipelines.
EXTENDED_STATISTICS: list[Statistic] = [s for s in BUILTIN_STATISTICS if s.name in EXTENDED_STATISTIC_NAMES]

#: *Diagnostic* statistics: all 45 built-in statistics.
#: Maximum detail; use when investigating data quality or distribution shape.
DIAGNOSTIC_STATISTICS: list[Statistic] = BUILTIN_STATISTICS

#: Mapping from tier name to the corresponding list of statistics.
#: Used by :class:`~parcellate.parcellation.volume.VolumetricParcellator`
#: when a ``stat_tier`` string is supplied instead of explicit functions.
#:
#: Valid keys: ``"core"``, ``"extended"``, ``"diagnostic"``, ``"all"``.
STATISTIC_TIERS: dict[str, list[Statistic]] = {
    "core": CORE_STATISTICS,
    "extended": EXTENDED_STATISTICS,
    "diagnostic": DIAGNOSTIC_STATISTICS,
    "all": BUILTIN_STATISTICS,
}
