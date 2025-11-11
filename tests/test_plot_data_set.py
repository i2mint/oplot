"""Tests for oplot.plot_data_set module"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from oplot.plot_data_set import (
    density_distribution,
    scatter_and_color_according_to_y,
    side_by_side_bar,
)


class TestDensityDistribution:
    """Tests for density_distribution function"""

    def test_density_distribution_basic(self):
        """Test density_distribution with basic input"""
        data_dict = {
            'dist1': np.random.normal(0, 1, 100),
            'dist2': np.random.normal(5, 2, 100),
        }
        density_distribution(data_dict)
        plt.close('all')

    def test_density_distribution_with_ax(self):
        """Test density_distribution with provided axes"""
        data_dict = {
            'dist1': np.random.normal(0, 1, 100),
            'dist2': np.random.normal(5, 2, 100),
        }
        fig, ax = plt.subplots()
        density_distribution(data_dict, ax=ax)
        assert len(ax.lines) > 0  # Should have plotted lines
        plt.close(fig)

    def test_density_distribution_with_custom_colors(self):
        """Test density_distribution with custom colors"""
        data_dict = {
            'dist1': np.random.normal(0, 1, 100),
            'dist2': np.random.normal(5, 2, 100),
        }
        density_distribution(data_dict, colors=('red', 'blue'))
        plt.close('all')

    def test_density_distribution_without_location_text(self):
        """Test density_distribution without location text"""
        data_dict = {
            'dist1': np.random.normal(0, 1, 100),
        }
        density_distribution(data_dict, display_location_text=False)
        plt.close('all')

    def test_density_distribution_with_list_input(self):
        """Test density_distribution with list input (converted to dict)"""
        data_list = [
            np.random.normal(0, 1, 100),
            np.random.normal(5, 2, 100),
        ]
        density_distribution(data_list)
        plt.close('all')


class TestScatterAndColorAccordingToY:
    """Tests for scatter_and_color_according_to_y function"""

    def test_scatter_2d_with_y(self):
        """Test 2D scatter plot with y labels"""
        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = np.array([0] * 25 + [1] * 25)
        scatter_and_color_according_to_y(X, y, projection='2d', dim_reduct='PCA')
        plt.close('all')

    def test_scatter_2d_without_y(self):
        """Test 2D scatter plot without y labels"""
        np.random.seed(42)
        X = np.random.rand(50, 5)
        scatter_and_color_according_to_y(X, projection='2d')
        plt.close('all')

    def test_scatter_3d_with_pca(self):
        """Test 3D scatter plot with PCA"""
        np.random.seed(42)
        X = np.random.rand(50, 10)
        y = np.array([0, 1, 2] * 16 + [0, 1])
        scatter_and_color_according_to_y(X, y, projection='3d', dim_reduct='PCA')
        plt.close('all')

    def test_scatter_with_lda_multiclass(self):
        """Test scatter plot with LDA and multiple classes"""
        np.random.seed(42)
        X = np.random.rand(60, 10)
        y = np.array([0, 1, 2] * 20)
        scatter_and_color_according_to_y(X, y, projection='2d', dim_reduct='LDA')
        plt.close('all')

    def test_scatter_with_low_dimensional_data(self):
        """Test scatter plot when data already has target dimensions"""
        np.random.seed(42)
        X = np.random.rand(50, 2)  # Already 2D
        y = np.array([0] * 25 + [1] * 25)
        scatter_and_color_according_to_y(X, y, projection='2d')
        plt.close('all')

    def test_scatter_with_float_y(self):
        """Test scatter plot with continuous y values"""
        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = np.random.rand(50)
        scatter_and_color_according_to_y(X, y, projection='2d', dim_reduct='PCA')
        plt.close('all')

    def test_scatter_1d(self):
        """Test 1D scatter plot"""
        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = np.array([0] * 25 + [1] * 25)
        scatter_and_color_according_to_y(X, y, projection='1d', dim_reduct='PCA')
        plt.close('all')


class TestSideBySideBar:
    """Tests for side_by_side_bar function"""

    def test_side_by_side_bar_basic(self):
        """Test side_by_side_bar with basic input"""
        list_of_values = [[1, 2, 3], [4, 5, 6]]
        side_by_side_bar(list_of_values)
        plt.close('all')

    def test_side_by_side_bar_with_names(self):
        """Test side_by_side_bar with custom names"""
        list_of_values = [[1, 2, 3], [4, 5, 6]]
        side_by_side_bar(list_of_values, list_names=['Group A', 'Group B'])
        plt.close('all')

    def test_side_by_side_bar_with_colors(self):
        """Test side_by_side_bar with custom colors"""
        list_of_values = [[1, 2, 3], [4, 5, 6]]
        side_by_side_bar(list_of_values, colors=['red', 'blue'])
        plt.close('all')

    def test_side_by_side_bar_with_custom_width(self):
        """Test side_by_side_bar with custom width and spacing"""
        list_of_values = [[1, 2, 3], [4, 5, 6]]
        side_by_side_bar(list_of_values, width=0.5, spacing=2)
        plt.close('all')
