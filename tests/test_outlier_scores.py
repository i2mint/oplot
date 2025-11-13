"""Tests for oplot.outlier_scores module"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from oplot.outlier_scores import (
    plot_scores_and_zones,
    sort_scores_truth,
    get_percentiles,
    get_confusion_zones_std,
)


class TestPlotScoresAndZones:
    """Tests for plot_scores_and_zones function"""

    def test_plot_scores_and_zones_basic(self):
        """Test plot_scores_and_zones with basic input"""
        scores = np.random.rand(100)
        zones = [0.3, 0.7, 0.9]
        plot_scores_and_zones(scores, zones)
        plt.close('all')

    def test_plot_scores_and_zones_with_title(self):
        """Test plot_scores_and_zones with title"""
        scores = np.random.rand(100)
        zones = [0.25, 0.5, 0.75]
        plot_scores_and_zones(scores, zones, title='Test Plot')
        plt.close('all')

    def test_plot_scores_and_zones_without_lines(self):
        """Test plot_scores_and_zones without zone lines"""
        scores = np.random.rand(100)
        zones = [0.3, 0.7]
        plot_scores_and_zones(scores, zones, lines=False)
        plt.close('all')

    def test_plot_scores_and_zones_with_box(self):
        """Test plot_scores_and_zones with custom box limits"""
        scores = np.random.rand(100)
        zones = [0.3, 0.7, 0.9]
        plot_scores_and_zones(scores, zones, box=(0, 100, 0, 1))
        plt.close('all')


class TestSortScoresTruth:
    """Tests for sort_scores_truth function"""

    def test_sort_scores_truth_basic(self):
        """Test sort_scores_truth with basic input"""
        scores = np.array([0.3, 0.1, 0.9, 0.5])
        truth = np.array([0, 1, 1, 0])
        sorted_scores, sorted_truth = sort_scores_truth(scores, truth)

        # Check that scores are sorted
        assert np.all(sorted_scores[:-1] <= sorted_scores[1:])
        # Check that truth array is aligned
        assert len(sorted_scores) == len(sorted_truth)

    def test_sort_scores_truth_preserves_alignment(self):
        """Test that sort_scores_truth preserves score-truth alignment"""
        scores = np.array([0.8, 0.2, 0.5])
        truth = np.array([1, 0, 1])
        sorted_scores, sorted_truth = sort_scores_truth(scores, truth)

        # Manually verify alignment
        # 0.2 (index 1) -> truth 0
        # 0.5 (index 2) -> truth 1
        # 0.8 (index 0) -> truth 1
        expected_truth = np.array([0, 1, 1])
        np.testing.assert_array_equal(sorted_truth, expected_truth)


class TestGetPercentiles:
    """Tests for get_percentiles function"""

    def test_get_percentiles_basic(self):
        """Test get_percentiles with basic input"""
        arr = [1, 2, 3, 4]
        result = get_percentiles(arr, n_percentiles=2)
        assert len(result) == 2
        assert result[0] <= result[1]

    def test_get_percentiles_doctest_examples(self):
        """Test get_percentiles matches doctest examples"""
        arr = [1, 2, 3, 4]

        # Test from docstring
        np.testing.assert_array_equal(get_percentiles(arr, n_percentiles=1), [3])
        np.testing.assert_array_equal(get_percentiles(arr, n_percentiles=2), [2, 3])
        np.testing.assert_array_equal(get_percentiles(arr, n_percentiles=3), [2, 3, 4])

    def test_get_percentiles_with_interpolation(self):
        """Test get_percentiles when interpolation is needed"""
        arr = [1, 2, 3, 4]
        result = get_percentiles(arr, n_percentiles=5)
        assert len(result) == 5
        # Should be interpolated values
        assert result[0] > arr[0]


class TestGetConfusionZonesStd:
    """Tests for get_confusion_zones_std function"""

    def test_get_confusion_zones_std_basic(self):
        """Test get_confusion_zones_std with basic input"""
        scores = np.random.normal(0, 1, 100)
        truth = np.array([0] * 50 + [1] * 50)
        zones = get_confusion_zones_std(scores, truth, n_zones=5)

        assert len(zones) == 5
        # Zones should be increasing
        assert np.all(zones[:-1] <= zones[1:])

    def test_get_confusion_zones_std_without_truth(self):
        """Test get_confusion_zones_std without truth array"""
        scores = np.random.normal(0, 1, 100)
        zones = get_confusion_zones_std(scores, n_zones=4)

        assert len(zones) == 4
        assert np.all(zones[:-1] <= zones[1:])

    def test_get_confusion_zones_std_custom_std_per_zone(self):
        """Test get_confusion_zones_std with custom std_per_zone"""
        scores = np.random.normal(0, 1, 100)
        zones = get_confusion_zones_std(scores, n_zones=3, std_per_zone=1.0)

        assert len(zones) == 3
