"""Plots that are frequently useful to us"""

from oplot.multiplots import ax_func_to_plot
from oplot.ui_scores_mapping import make_ui_score_mapping
from oplot.outlier_scores import plot_scores_and_zones
from oplot.plot_data_set import scatter_and_color_according_to_y, side_by_side_bar
from oplot.plot_stats import plot_confusion_matrix
from oplot.matrix import (
    heatmap,
    heatmap_sns,
    xy_boxplot,
    vlines_ranges,
    vlines_of_matrix,
    labeled_heatmap,
    get_figsize_to_fit,
    plot_simil_mat_with_labels,
    hierarchical_cluster_sorted_heatmap,
)
