"""Tests for oplot.matrix module"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from oplot.matrix import (
    heatmap,
    xy_boxplot,
    vlines_ranges,
    vlines_of_matrix,
)


class TestHeatmap:
    """Tests for heatmap function"""

    def test_heatmap_with_array(self):
        """Test heatmap with numpy array input"""
        data = np.random.rand(5, 5)
        fig, ax = plt.subplots()
        heatmap(data, ax=ax)
        plt.close(fig)

    def test_heatmap_with_dataframe(self):
        """Test heatmap with pandas DataFrame input"""
        data = pd.DataFrame(np.random.rand(5, 5), columns=list('ABCDE'))
        fig, ax = plt.subplots()
        heatmap(data, ax=ax)
        plt.close(fig)

    def test_heatmap_with_labels(self):
        """Test heatmap with custom labels"""
        data = np.random.rand(3, 3)
        fig, ax = plt.subplots()
        heatmap(data, col_labels=['A', 'B', 'C'], ax=ax)
        plt.close(fig)

    def test_heatmap_creates_figure_when_ax_none(self):
        """Test that heatmap creates a figure when ax is None"""
        data = np.random.rand(3, 3)
        heatmap(data)
        plt.close('all')

    def test_heatmap_with_custom_figsize(self):
        """Test heatmap with custom figure size"""
        data = np.random.rand(4, 4)
        heatmap(data, figsize=(8, 6))
        plt.close('all')

    def test_heatmap_with_return_gcf(self):
        """Test heatmap returns figure when return_gcf=True"""
        data = np.random.rand(4, 4)
        fig = heatmap(data, return_gcf=True)
        assert fig is not None
        plt.close(fig)


class TestXyBoxplot:
    """Tests for xy_boxplot function"""

    def test_xy_boxplot_without_y(self):
        """Test xy_boxplot without y parameter"""
        X = np.random.rand(20, 3)
        xy_boxplot(X)
        plt.close('all')

    def test_xy_boxplot_with_y(self):
        """Test xy_boxplot with y parameter"""
        X = np.random.rand(20, 3)
        y = np.array([0] * 10 + [1] * 10)
        xy_boxplot(X, y=y)
        plt.close('all')

    def test_xy_boxplot_with_col_labels(self):
        """Test xy_boxplot with column labels"""
        X = np.random.rand(20, 3)
        xy_boxplot(X, col_labels=['A', 'B', 'C'])
        plt.close('all')


class TestVlinesRanges:
    """Tests for vlines_ranges function"""

    def test_vlines_ranges_default(self):
        """Test vlines_ranges with default parameters"""
        X = np.random.rand(10, 5)
        vlines_ranges(X)
        plt.close('all')

    def test_vlines_ranges_with_aggr_int(self):
        """Test vlines_ranges with integer aggr parameter"""
        X = np.random.rand(10, 5)
        vlines_ranges(X, aggr=2)
        plt.close('all')

    def test_vlines_ranges_with_custom_aggr(self):
        """Test vlines_ranges with custom aggregation functions"""
        X = np.random.rand(10, 5)
        vlines_ranges(X, aggr=('min', 'mean', 'max'))
        plt.close('all')


class TestVlinesOfMatrix:
    """Tests for vlines_of_matrix function"""

    def test_vlines_of_matrix_basic(self):
        """Test vlines_of_matrix with basic input"""
        X = np.random.rand(10, 5)
        vlines_of_matrix(X)
        plt.close('all')

    def test_vlines_of_matrix_with_col_labels(self):
        """Test vlines_of_matrix with column labels"""
        X = np.random.rand(10, 3)
        vlines_of_matrix(X, col_labels=['A', 'B', 'C'])
        plt.close('all')

    def test_vlines_of_matrix_with_figsize(self):
        """Test vlines_of_matrix with custom figure size"""
        X = np.random.rand(10, 3)
        vlines_of_matrix(X, figsize=(10, 6))
        plt.close('all')

    def test_vlines_of_matrix_with_ax(self):
        """Test vlines_of_matrix with provided axes"""
        X = np.random.rand(10, 3)
        fig, ax = plt.subplots()
        vlines_of_matrix(X, ax=ax)
        plt.close(fig)
