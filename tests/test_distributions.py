"""Tests for oplot.distributions module"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from oplot.distributions import kdeplot_w_boundary_condition


class TestKdeplotWBoundaryCondition:
    """Tests for kdeplot_w_boundary_condition function"""

    def test_kdeplot_without_boundary(self):
        """Test kdeplot without boundary condition (falls back to seaborn)"""
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100)
        })
        ax = kdeplot_w_boundary_condition(
            data=data, x='x', y='y', boundary_condition=None
        )
        assert isinstance(ax, Axes)
        plt.close('all')

    def test_kdeplot_with_boundary_condition(self):
        """Test kdeplot with boundary condition y <= x"""
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100)
        })
        boundary_condition = lambda X, Y: Y <= X
        ax = kdeplot_w_boundary_condition(
            data=data, x='x', y='y', boundary_condition=boundary_condition
        )
        assert isinstance(ax, Axes)
        plt.close('all')

    def test_kdeplot_with_custom_cmap(self):
        """Test kdeplot with custom colormap"""
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100)
        })
        boundary_condition = lambda X, Y: Y <= X
        ax = kdeplot_w_boundary_condition(
            data=data,
            x='x',
            y='y',
            boundary_condition=boundary_condition,
            cmap='viridis'
        )
        assert isinstance(ax, Axes)
        plt.close('all')

    def test_kdeplot_with_figsize(self):
        """Test kdeplot with custom figure size"""
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100)
        })
        ax = kdeplot_w_boundary_condition(
            data=data, x='x', y='y', figsize=(8, 6)
        )
        assert isinstance(ax, Axes)
        plt.close('all')

    def test_kdeplot_with_provided_ax(self):
        """Test kdeplot with provided axes"""
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100)
        })
        fig, ax = plt.subplots()
        result_ax = kdeplot_w_boundary_condition(data=data, x='x', y='y', ax=ax)
        assert result_ax is ax
        plt.close(fig)
