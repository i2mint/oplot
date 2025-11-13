"""Tests for oplot.plot_mappings module"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from oplot.plot_mappings import dict_bar_plot


class TestDictBarPlot:
    """Tests for dict_bar_plot function"""

    def test_dict_bar_plot_basic(self):
        """Test dict_bar_plot with basic dictionary"""
        d = {'A': 10, 'B': 20, 'C': 15}
        dict_bar_plot(d)
        plt.close('all')

    def test_dict_bar_plot_with_title(self):
        """Test dict_bar_plot with title"""
        d = {'A': 10, 'B': 20, 'C': 15}
        dict_bar_plot(d, title='Test Bar Plot')
        plt.close('all')

    def test_dict_bar_plot_with_labels(self):
        """Test dict_bar_plot with custom labels"""
        d = {'A': 10, 'B': 20, 'C': 15}
        dict_bar_plot(d, xlabel='Categories', ylabel='Values')
        plt.close('all')

    def test_dict_bar_plot_with_figsize(self):
        """Test dict_bar_plot with custom figure size"""
        d = {'A': 10, 'B': 20, 'C': 15, 'D': 25}
        dict_bar_plot(d, figsize=(10, 6))
        plt.close('all')

    def test_dict_bar_plot_empty_dict(self):
        """Test dict_bar_plot with empty dictionary"""
        d = {}
        # This may raise an error or create empty plot - both are acceptable
        try:
            dict_bar_plot(d)
            plt.close('all')
        except Exception:
            pass  # Empty dict might raise an error, which is fine
