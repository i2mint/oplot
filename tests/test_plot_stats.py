"""Tests for oplot.plot_stats module"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from oplot.plot_stats import (
    plot_confusion_matrix,
    make_tables_tn_fp_fn_tp,
    make_normal_outlier_timeline,
    render_mpl_table,
)


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function"""

    def test_plot_confusion_matrix_basic(self):
        """Test plot_confusion_matrix with basic input"""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        plot_confusion_matrix(y_true, y_pred)
        plt.close('all')

    def test_plot_confusion_matrix_multiclass(self):
        """Test plot_confusion_matrix with multiple classes"""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 1, 2, 0, 2, 2])
        plot_confusion_matrix(y_true, y_pred)
        plt.close('all')

    def test_plot_confusion_matrix_normalized(self):
        """Test plot_confusion_matrix with normalization"""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        plot_confusion_matrix(y_true, y_pred, normalize=True)
        plt.close('all')

    def test_plot_confusion_matrix_with_ax(self):
        """Test plot_confusion_matrix with provided axes"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        fig, ax = plt.subplots()
        plot_confusion_matrix(y_true, y_pred, ax=ax)
        plt.close(fig)

    def test_plot_confusion_matrix_with_custom_classes(self):
        """Test plot_confusion_matrix with custom class labels"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        plot_confusion_matrix(y_true, y_pred, classes=[0, 1])
        plt.close('all')


class TestMakeTablesTnFpFnTp:
    """Tests for make_tables_tn_fp_fn_tp function"""

    def test_make_tables_basic(self):
        """Test make_tables_tn_fp_fn_tp with basic input"""
        truth = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
        df = make_tables_tn_fp_fn_tp(truth, scores, n_thresholds=3)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'Threshold' in df.columns

    def test_make_tables_with_custom_range(self):
        """Test make_tables_tn_fp_fn_tp with custom threshold range"""
        truth = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
        df = make_tables_tn_fp_fn_tp(
            truth, scores, threshold_range=(0.2, 0.7), n_thresholds=3
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_make_tables_normalized(self):
        """Test make_tables_tn_fp_fn_tp with normalization"""
        truth = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
        df = make_tables_tn_fp_fn_tp(truth, scores, n_thresholds=3, normalize=True)
        assert isinstance(df, pd.DataFrame)
        # Check that values are between 0 and 1 when normalized
        assert df['True Positive'].max() <= 1.0


class TestMakeNormalOutlierTimeline:
    """Tests for make_normal_outlier_timeline function"""

    def test_make_normal_outlier_timeline_basic(self):
        """Test make_normal_outlier_timeline with basic input"""
        scores = np.random.rand(100)
        y = np.array(['normal'] * 50 + ['outlier'] * 50)
        make_normal_outlier_timeline(y, scores)
        plt.close('all')

    def test_make_normal_outlier_timeline_with_y_order(self):
        """Test make_normal_outlier_timeline with specified y_order"""
        scores = np.random.rand(90)
        y = np.array(['A'] * 30 + ['B'] * 30 + ['C'] * 30)
        make_normal_outlier_timeline(y, scores, y_order=['C', 'B', 'A'])
        plt.close('all')

    def test_make_normal_outlier_timeline_with_custom_figsize(self):
        """Test make_normal_outlier_timeline with custom figure size"""
        scores = np.random.rand(60)
        y = np.array(['normal'] * 30 + ['outlier'] * 30)
        make_normal_outlier_timeline(y, scores, fig_size=(12, 4))
        plt.close('all')

    def test_make_normal_outlier_timeline_preserves_order(self):
        """Test that make_normal_outlier_timeline preserves insertion order when y_order=None"""
        # This tests the fix for Issue #6
        scores = np.array([1, 2, 3, 4, 5, 6])
        y = np.array(['C', 'A', 'B', 'C', 'A', 'B'])
        # When y_order is None, should preserve order of first appearance: C, A, B
        # NOT alphabetical order: A, B, C
        make_normal_outlier_timeline(y, scores, y_order=None)
        plt.close('all')


class TestRenderMplTable:
    """Tests for render_mpl_table function"""

    def test_render_mpl_table_basic(self):
        """Test render_mpl_table with basic DataFrame"""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        render_mpl_table(df)
        plt.close('all')

    def test_render_mpl_table_with_rounding(self):
        """Test render_mpl_table with decimal rounding"""
        df = pd.DataFrame({'A': [1.123456, 2.789], 'B': [3.456789, 4.123]})
        render_mpl_table(df, round_decimals=2)
        plt.close('all')
