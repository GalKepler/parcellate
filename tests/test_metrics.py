"""Tests for parcellate.metrics.volume module."""

from __future__ import annotations

import numpy as np
import pytest

from parcellate.metrics.volume import (
    _iqr_filter,
    _robust_filter,
    _z_score_filter,
    abs_excess_kurtosis,
    abs_skewness,
    bimodality_coefficient,
    cv,
    dagostino_k2,
    dagostino_p,
    excess_tail_mass,
    histogram_entropy,
    iqr_filtered_mean,
    iqr_filtered_std,
    is_bimodal,
    is_heavy_tailed,
    is_strongly_skewed,
    left_tail_mass,
    log_dagostino_k2,
    mad_median,
    percentile_5,
    percentile_25,
    percentile_50,
    percentile_75,
    percentile_95,
    prop_outliers_2sd,
    prop_outliers_3sd,
    prop_outliers_iqr,
    qq_correlation,
    qq_r_squared,
    right_tail_mass,
    robust_cv,
    robust_mean,
    robust_std,
    shapiro_p,
    shapiro_w,
    skewness,
    tail_asymmetry,
    volsum,
    voxel_count,
    z_filtered_mean,
    z_filtered_std,
)
from parcellate.metrics.volume import (
    excess_kurtosis as kurtosis,
)


class TestZScoreFilter:
    """Tests for _z_score_filter helper function."""

    def test_filters_outliers(self) -> None:
        # Values with clear outliers - need many normal values for z-score to work
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1000.0])
        filtered = _z_score_filter(values, z_thresh=2.0)
        assert 1000.0 not in filtered
        assert len(filtered) == 10

    def test_returns_all_when_no_outliers(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        filtered = _z_score_filter(values, z_thresh=3.0)
        assert len(filtered) == 5

    def test_returns_original_when_std_zero(self) -> None:
        values = np.array([5.0, 5.0, 5.0, 5.0])
        filtered = _z_score_filter(values, z_thresh=3.0)
        np.testing.assert_array_equal(filtered, values)

    def test_handles_nan_values(self) -> None:
        values = np.array([1.0, 2.0, np.nan, 3.0])
        filtered = _z_score_filter(values, z_thresh=3.0)
        # nanmean and nanstd ignore NaN, so the result depends on the z-scores
        assert len(filtered) <= len(values)


class TestIQRFilter:
    """Tests for _iqr_filter helper function."""

    def test_filters_outliers(self) -> None:
        # Normal distribution with outliers
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -100.0])
        filtered = _iqr_filter(values, factor=1.5)
        assert 100.0 not in filtered
        assert -100.0 not in filtered

    def test_custom_factor(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
        # With factor=0.5, more values get filtered
        filtered_strict = _iqr_filter(values, factor=0.5)
        # With factor=3.0, fewer values get filtered
        filtered_loose = _iqr_filter(values, factor=3.0)
        assert len(filtered_strict) <= len(filtered_loose)

    def test_returns_all_when_no_outliers(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        filtered = _iqr_filter(values, factor=1.5)
        assert len(filtered) == 5


class TestRobustFilter:
    """Tests for _robust_filter helper function."""

    def test_filters_outliers(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        filtered = _robust_filter(values, threshold=3.5)
        assert 100.0 not in filtered

    def test_returns_original_when_mad_zero(self) -> None:
        values = np.array([5.0, 5.0, 5.0, 5.0])
        filtered = _robust_filter(values, threshold=3.5)
        np.testing.assert_array_equal(filtered, values)

    def test_custom_threshold(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 8.0])
        # Lower threshold is more strict
        filtered_strict = _robust_filter(values, threshold=1.5)
        filtered_loose = _robust_filter(values, threshold=5.0)
        assert len(filtered_strict) <= len(filtered_loose)


class TestZFilteredMean:
    """Tests for z_filtered_mean function."""

    def test_computes_mean_after_filtering(self) -> None:
        # Use many values so outlier has high z-score
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1000.0])
        result = z_filtered_mean(values, z_thresh=2.0)
        # Mean of [1-10] = 5.5
        assert result == pytest.approx(5.5)

    def test_returns_original_mean_when_all_filtered(self) -> None:
        # Edge case: very strict threshold filters everything
        values = np.array([1.0, 100.0])
        result = z_filtered_mean(values, z_thresh=0.01)
        # Falls back to original mean
        assert result == pytest.approx(np.nanmean(values))

    def test_identical_values(self) -> None:
        values = np.array([5.0, 5.0, 5.0])
        result = z_filtered_mean(values)
        assert result == pytest.approx(5.0)


class TestZFilteredStd:
    """Tests for z_filtered_std function."""

    def test_computes_std_after_filtering(self) -> None:
        # Use many values so outlier has high z-score
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1000.0])
        result = z_filtered_std(values, z_thresh=2.0)
        expected = np.nanstd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        assert result == pytest.approx(expected)

    def test_returns_zero_for_identical_values(self) -> None:
        values = np.array([5.0, 5.0, 5.0])
        result = z_filtered_std(values)
        assert result == 0.0

    def test_returns_original_std_when_all_filtered(self) -> None:
        values = np.array([1.0, 100.0])
        result = z_filtered_std(values, z_thresh=0.01)
        # Falls back to original std
        assert result == pytest.approx(np.nanstd(values))


class TestIQRFilteredMean:
    """Tests for iqr_filtered_mean function."""

    def test_computes_mean_after_filtering(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = iqr_filtered_mean(values, factor=1.5)
        # Mean without the outlier
        expected = np.nanmean([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result == pytest.approx(expected)

    def test_returns_original_mean_when_empty_filter(self) -> None:
        # Create case where IQR filter removes everything
        values = np.array([1.0, 1.0, 1.0, 1.0])  # IQR = 0
        result = iqr_filtered_mean(values, factor=0.0)  # Very strict
        # Should return original mean
        assert result == pytest.approx(1.0)


class TestIQRFilteredStd:
    """Tests for iqr_filtered_std function."""

    def test_computes_std_after_filtering(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = iqr_filtered_std(values, factor=1.5)
        expected = np.nanstd([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result == pytest.approx(expected)

    def test_returns_original_std_when_empty_filter(self) -> None:
        values = np.array([1.0, 1.0, 1.0, 1.0])
        result = iqr_filtered_std(values, factor=0.0)
        assert result == pytest.approx(np.nanstd(values))


class TestRobustMean:
    """Tests for robust_mean function."""

    def test_computes_mean_after_filtering(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = robust_mean(values)
        # Should exclude the outlier
        assert result < 20.0  # Much less than if 100 was included

    def test_returns_median_when_mad_zero(self) -> None:
        values = np.array([5.0, 5.0, 5.0])
        result = robust_mean(values)
        assert result == pytest.approx(5.0)

    def test_returns_median_when_empty_filter(self) -> None:
        # This is hard to trigger, but we test the fallback path
        values = np.array([1.0, 2.0])
        result = robust_mean(values)
        assert isinstance(result, float)


class TestRobustStd:
    """Tests for robust_std function."""

    def test_computes_std_after_filtering(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = robust_std(values)
        # Should be smaller than if 100 was included
        assert result < np.nanstd(values)

    def test_returns_original_std_when_mad_zero(self) -> None:
        values = np.array([5.0, 5.0, 5.0])
        result = robust_std(values)
        assert result == pytest.approx(0.0)


class TestMadMedian:
    """Tests for mad_median function."""

    def test_computes_mad(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mad_median(values)
        # Median is 3, deviations are [2, 1, 0, 1, 2], MAD = 1
        assert result == pytest.approx(1.0)

    def test_mad_of_identical_values_is_zero(self) -> None:
        values = np.array([5.0, 5.0, 5.0])
        result = mad_median(values)
        assert result == 0.0

    def test_handles_nan(self) -> None:
        values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = mad_median(values)
        # Should compute MAD ignoring NaN
        assert not np.isnan(result)


class TestVolsum:
    """Tests for volsum function."""

    def test_computes_sum(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = volsum(values)
        assert result == pytest.approx(15.0)

    def test_handles_nan(self) -> None:
        values = np.array([1.0, 2.0, np.nan, 4.0])
        result = volsum(values)
        assert result == pytest.approx(7.0)

    def test_all_nan_returns_zero(self) -> None:
        values = np.array([np.nan, np.nan])
        result = volsum(values)
        assert result == 0.0


class TestVoxelCount:
    """Tests for voxel_count function."""

    def test_counts_true_values(self) -> None:
        mask = np.array([True, True, False, True, False])
        result = voxel_count(mask)
        assert result == 3

    def test_counts_nonzero_as_true(self) -> None:
        mask = np.array([1, 2, 0, 3, 0])
        result = voxel_count(mask)
        assert result == 3

    def test_empty_mask(self) -> None:
        mask = np.array([False, False, False])
        result = voxel_count(mask)
        assert result == 0

    def test_all_true(self) -> None:
        mask = np.array([True, True, True])
        result = voxel_count(mask)
        assert result == 3


class TestSkewness:
    """Tests for skewness function."""

    def test_symmetric_distribution_zero_skew(self) -> None:
        """Test symmetric distribution has near-zero skewness."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = skewness(values)
        assert abs(result) < 0.1  # Should be close to 0 for symmetric data

    def test_positive_skew(self) -> None:
        """Test right-skewed distribution has positive skewness."""
        values = np.array([1.0, 1.0, 1.0, 1.0, 10.0])  # Long right tail
        result = skewness(values)
        assert result > 0

    def test_negative_skew(self) -> None:
        """Test left-skewed distribution has negative skewness."""
        values = np.array([10.0, 10.0, 10.0, 10.0, 1.0])  # Long left tail
        result = skewness(values)
        assert result < 0

    def test_constant_values_nan(self) -> None:
        """Test constant values returns NaN."""
        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = skewness(values)
        assert np.isnan(result)

    def test_handles_nan_values(self) -> None:
        """Test skewness handles NaN values."""
        values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
        result = skewness(values)
        assert not np.isnan(result)  # Should compute on non-NaN values

    def test_all_nan_returns_nan(self) -> None:
        """Test all NaN values returns NaN."""
        values = np.array([np.nan, np.nan, np.nan])
        result = skewness(values)
        assert np.isnan(result)

    def test_normal_distribution_low_skew(self) -> None:
        """Test normal distribution has low skewness."""
        np.random.seed(42)
        values = np.random.normal(0, 1, 1000)
        result = skewness(values)
        assert abs(result) < 0.2  # Should be close to 0


class TestKurtosis:
    """Tests for kurtosis function."""

    def test_normal_distribution_near_zero(self) -> None:
        """Test normal distribution has near-zero excess kurtosis."""
        np.random.seed(42)
        values = np.random.normal(0, 1, 1000)
        result = kurtosis(values)
        assert abs(result) < 0.5  # Excess kurtosis should be ~0 for normal

    def test_heavy_tails_positive_kurtosis(self) -> None:
        """Test heavy-tailed distribution has positive kurtosis."""
        # Create distribution with outliers (heavy tails)
        values = np.array([0.0] * 90 + [10.0] * 5 + [-10.0] * 5)
        result = kurtosis(values)
        assert result > 0

    def test_light_tails_negative_kurtosis(self) -> None:
        """Test uniform distribution has negative kurtosis."""
        values = np.linspace(0, 10, 100)  # Uniform-like
        result = kurtosis(values)
        assert result < 0

    def test_constant_values_nan(self) -> None:
        """Test constant values returns NaN."""
        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = kurtosis(values)
        assert np.isnan(result)

    def test_handles_nan_values(self) -> None:
        """Test kurtosis handles NaN values."""
        values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = kurtosis(values)
        assert not np.isnan(result)

    def test_all_nan_returns_nan(self) -> None:
        """Test all NaN values returns NaN."""
        values = np.array([np.nan, np.nan, np.nan])
        result = kurtosis(values)
        assert np.isnan(result)

    def test_minimum_sample_size(self) -> None:
        """Test with minimum sample size."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = kurtosis(values)
        assert isinstance(result, float)


class TestCVAndRobustCV:
    """Tests for cv and robust_cv functions."""

    def test_cv_normal(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cv(values)
        assert result == pytest.approx(np.std(values) / (np.mean(values) + 1e-10))

    def test_cv_near_zero_mean(self) -> None:
        """Test CV does not raise when mean is near zero."""
        values = np.array([0.001, 0.002, 0.003])
        result = cv(values)
        assert isinstance(result, float)

    def test_robust_cv_basic(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = robust_cv(values)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_robust_cv_near_zero_median(self) -> None:
        values = np.array([0.0, 0.001, 0.002])
        result = robust_cv(values)
        assert isinstance(result, float)


class TestQuartileDispersion:
    """Tests for quartile_dispersion function (including empty-array guard)."""

    def test_basic(self) -> None:
        from parcellate.metrics.volume import quartile_dispersion

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = quartile_dispersion(values)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_identical_values(self) -> None:
        from parcellate.metrics.volume import quartile_dispersion

        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = quartile_dispersion(values)
        assert result == pytest.approx(0.0)

    def test_empty_array_returns_nan(self) -> None:
        """Empty array must not raise IndexError (NumPy ≥ 2.0 regression)."""
        from parcellate.metrics.volume import quartile_dispersion

        result = quartile_dispersion(np.array([]))
        assert np.isnan(result)

    def test_all_nan_returns_nan(self) -> None:
        from parcellate.metrics.volume import quartile_dispersion

        result = quartile_dispersion(np.array([np.nan, np.nan]))
        assert np.isnan(result)


class TestAbsSkewnessAndKurtosis:
    """Tests for abs_skewness and abs_excess_kurtosis."""

    def test_abs_skewness_is_nonnegative(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = abs_skewness(values)
        assert result >= 0.0

    def test_abs_skewness_symmetric(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs_skewness(values) == pytest.approx(0.0, abs=0.1)

    def test_abs_excess_kurtosis_is_nonnegative(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = abs_excess_kurtosis(values)
        assert result >= 0.0


class TestBimodalityCoefficient:
    """Tests for bimodality_coefficient."""

    def test_small_sample_returns_nan(self) -> None:
        values = np.array([1.0, 2.0, 3.0])  # n < 4
        assert np.isnan(bimodality_coefficient(values))

    def test_unimodal_below_threshold(self) -> None:
        np.random.seed(0)
        values = np.random.normal(0, 1, 200)
        result = bimodality_coefficient(values)
        assert isinstance(result, float)
        # Normal distribution typically has BC < 0.555
        assert result < 0.555

    def test_high_skew_above_threshold(self) -> None:
        # Highly skewed distribution gives BC > 0.555
        values = np.concatenate([np.zeros(95), np.array([100.0] * 5)])
        result = bimodality_coefficient(values)
        assert result > 0.555


class TestPropOutliers:
    """Tests for prop_outliers_2sd, prop_outliers_3sd, and prop_outliers_iqr."""

    def test_2sd_no_outliers(self) -> None:
        np.random.seed(42)
        values = np.random.normal(0, 1, 1000)
        result = prop_outliers_2sd(values)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_2sd_with_outliers(self) -> None:
        values = np.concatenate([np.ones(90), np.array([100.0] * 10)])
        result = prop_outliers_2sd(values)
        assert result > 0.0

    def test_3sd_no_outliers(self) -> None:
        np.random.seed(42)
        values = np.random.normal(0, 1, 1000)
        result = prop_outliers_3sd(values)
        assert 0.0 <= result <= 0.02

    def test_iqr_no_outliers(self) -> None:
        np.random.seed(42)
        values = np.random.normal(0, 1, 1000)
        result = prop_outliers_iqr(values)
        assert 0.0 <= result <= 0.05

    def test_iqr_with_outliers(self) -> None:
        values = np.concatenate([np.ones(90), np.array([1000.0] * 10)])
        result = prop_outliers_iqr(values)
        assert result > 0.0

    def test_iqr_empty_returns_nan(self) -> None:
        """Empty array guard added in fix."""
        result = prop_outliers_iqr(np.array([]))
        assert np.isnan(result)


class TestTailMetrics:
    """Tests for left_tail_mass, right_tail_mass, tail_asymmetry, excess_tail_mass."""

    def test_left_tail_mass_symmetric(self) -> None:
        np.random.seed(0)
        values = np.random.normal(0, 1, 10000)
        result = left_tail_mass(values)
        assert pytest.approx(result, abs=0.01) == 0.023

    def test_right_tail_mass_symmetric(self) -> None:
        np.random.seed(0)
        values = np.random.normal(0, 1, 10000)
        result = right_tail_mass(values)
        assert pytest.approx(result, abs=0.01) == 0.023

    def test_tail_asymmetry_symmetric(self) -> None:
        np.random.seed(0)
        values = np.random.normal(0, 1, 10000)
        result = tail_asymmetry(values)
        assert abs(result) < 0.02

    def test_tail_asymmetry_right_skewed(self) -> None:
        values = np.concatenate([np.ones(80), np.array([10.0] * 20)])
        result = tail_asymmetry(values)
        assert result > 0.0

    def test_excess_tail_mass_near_normal(self) -> None:
        np.random.seed(0)
        values = np.random.normal(0, 1, 10000)
        result = excess_tail_mass(values)
        assert abs(result) < 0.02


class TestNormalityTests:
    """Tests for dagostino_k2/p, log_dagostino_k2, shapiro_w/p."""

    def test_dagostino_small_sample_nan(self) -> None:
        values = np.array([1.0, 2.0, 3.0])  # n < 20
        assert np.isnan(dagostino_k2(values))
        assert np.isnan(dagostino_p(values))
        assert np.isnan(log_dagostino_k2(values))

    def test_dagostino_normal_data(self) -> None:
        np.random.seed(42)
        values = np.random.normal(0, 1, 500)
        k2 = dagostino_k2(values)
        p = dagostino_p(values)
        assert k2 > 0
        assert 0.0 <= p <= 1.0

    def test_log_dagostino_is_log_of_k2(self) -> None:
        np.random.seed(42)
        values = np.random.normal(0, 1, 100)
        k2 = dagostino_k2(values)
        log_k2 = log_dagostino_k2(values)
        assert log_k2 == pytest.approx(np.log(k2 + 1e-10))

    def test_shapiro_small_sample_nan(self) -> None:
        values = np.array([1.0, 2.0])  # n < 3
        assert np.isnan(shapiro_w(values))
        assert np.isnan(shapiro_p(values))

    def test_shapiro_normal_data(self) -> None:
        np.random.seed(42)
        values = np.random.normal(0, 1, 100)
        w = shapiro_w(values)
        p = shapiro_p(values)
        assert 0.0 <= w <= 1.0
        assert 0.0 <= p <= 1.0


class TestQQMetrics:
    """Tests for qq_correlation and qq_r_squared."""

    def test_qq_correlation_normal(self) -> None:
        np.random.seed(42)
        values = np.random.normal(0, 1, 200)
        r = qq_correlation(values)
        assert r > 0.99  # Normal data should correlate very well

    def test_qq_r_squared_is_square_of_correlation(self) -> None:
        np.random.seed(42)
        values = np.random.normal(0, 1, 200)
        r = qq_correlation(values)
        r2 = qq_r_squared(values)
        assert r2 == pytest.approx(r**2)


class TestHistogramEntropy:
    """Tests for histogram_entropy."""

    def test_uniform_higher_entropy_than_single_point(self) -> None:
        # Single value has entropy = 0 (all mass in one bin)
        single = np.array([5.0] * 100)
        uniform = np.linspace(0, 10, 100)
        assert histogram_entropy(uniform) > histogram_entropy(single)

    def test_all_nan_returns_nan(self) -> None:
        values = np.array([np.nan, np.nan])
        assert np.isnan(histogram_entropy(values))


class TestPercentiles:
    """Tests for percentile_* functions."""

    def test_percentile_5(self) -> None:
        values = np.arange(1.0, 101.0)
        assert percentile_5(values) == pytest.approx(np.nanpercentile(values, 5))

    def test_percentile_25(self) -> None:
        values = np.arange(1.0, 101.0)
        assert percentile_25(values) == pytest.approx(25.75)

    def test_percentile_50_is_median(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert percentile_50(values) == pytest.approx(3.0)

    def test_percentile_75(self) -> None:
        values = np.arange(1.0, 101.0)
        assert percentile_75(values) == pytest.approx(np.nanpercentile(values, 75))

    def test_percentile_95(self) -> None:
        values = np.arange(1.0, 101.0)
        assert percentile_95(values) == pytest.approx(np.nanpercentile(values, 95))


class TestCategoricalFlags:
    """Tests for is_strongly_skewed, is_heavy_tailed, is_bimodal."""

    def test_is_strongly_skewed_true(self) -> None:
        values = np.concatenate([np.ones(90), np.array([100.0] * 10)])
        assert is_strongly_skewed(values)

    def test_is_strongly_skewed_false(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert not is_strongly_skewed(values)

    def test_is_heavy_tailed_true(self) -> None:
        values = np.array([0.0] * 80 + [50.0] * 10 + [-50.0] * 10)
        assert is_heavy_tailed(values)

    def test_is_bimodal_false_for_normal(self) -> None:
        np.random.seed(0)
        values = np.random.normal(0, 1, 500)
        assert not is_bimodal(values)
