"""Tests for oplot.multiplots module"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from oplot.multiplots import ax_func_to_plot


class TestAxFuncToPlot:
    """Tests for ax_func_to_plot function"""

    def test_ax_func_to_plot_basic(self):
        """Test ax_func_to_plot with basic input"""

        def plot_func(ax):
            ax.plot([1, 2, 3], [1, 4, 9])

        list_func = [plot_func] * 6
        ax_func_to_plot(list_func, n_per_row=3)
        plt.close('all')

    def test_ax_func_to_plot_with_labels(self):
        """Test ax_func_to_plot with axis labels"""

        def plot_func(ax):
            ax.plot([1, 2, 3], [1, 4, 9])

        list_func = [plot_func] * 4
        ax_func_to_plot(
            list_func,
            n_per_row=2,
            x_labels='X axis',
            y_labels='Y axis'
        )
        plt.close('all')

    def test_ax_func_to_plot_with_title(self):
        """Test ax_func_to_plot with title"""

        def plot_func(ax):
            ax.plot([1, 2, 3], [1, 4, 9])

        list_func = [plot_func] * 3
        ax_func_to_plot(
            list_func,
            n_per_row=3,
            title='Test Plot',
            title_font_size=12
        )
        plt.close('all')

    def test_ax_func_to_plot_custom_size(self):
        """Test ax_func_to_plot with custom size"""

        def plot_func(ax):
            ax.plot([1, 2, 3], [1, 4, 9])

        list_func = [plot_func] * 4
        ax_func_to_plot(
            list_func,
            n_per_row=2,
            width=10,
            height_row=5
        )
        plt.close('all')

    def test_ax_func_to_plot_outer_labels_only(self):
        """Test ax_func_to_plot with outer axis labels only"""

        def plot_func(ax):
            ax.plot([1, 2, 3], [1, 4, 9])

        list_func = [plot_func] * 6
        ax_func_to_plot(
            list_func,
            n_per_row=3,
            outer_axis_labels_only=True,
            x_labels='X',
            y_labels='Y'
        )
        plt.close('all')

    def test_ax_func_to_plot_various_list_sizes(self):
        """Test ax_func_to_plot with various list sizes"""

        def plot_func(ax):
            ax.scatter(np.random.rand(10), np.random.rand(10))

        for n_funcs in [1, 2, 3, 5, 7, 10]:
            list_func = [plot_func] * n_funcs
            ax_func_to_plot(list_func, n_per_row=3, plot=False)
            plt.close('all')
