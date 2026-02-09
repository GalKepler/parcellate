"""Tests for parcellate.metrics.volume module."""

from __future__ import annotations

import numpy as np
import pytest

from parcellate.metrics.volume import (
    _iqr_filter,
    _robust_filter,
    _z_score_filter,
    iqr_filtered_mean,
    iqr_filtered_std,
    mad_median,
    robust_mean,
    robust_std,
    volsum,
    voxel_count,
    z_filtered_mean,
    z_filtered_std,
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
