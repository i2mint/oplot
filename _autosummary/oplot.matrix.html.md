# oplot.matrix

Plotting matrices and matrix-structured data

### Functions

| [`get_figsize_to_fit`](#oplot.matrix.get_figsize_to_fit)(shape[, max_size])            | Calculate a proportional figsize based on the dimensions of the dataframe, with the larger dimension capped at `max_size`.                                                                                                                              |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`heatmap`](#oplot.matrix.heatmap)(X[, y, col_labels, figsize, cmap, ...])  | Heatmap plot of (X, y) sklearn-like data                                                                                                                                                                                                                |
| [`heatmap_sns`](#oplot.matrix.heatmap_sns)(df, \*[, cmap, xlabel, ylabel, ...]) | A reusable heatmap function to visualize numerical dataframes with customizable options, including adding vertical and horizontal lines for better readability.                                                                                         |
| [`hierarchical_cluster_sorted_heatmap`](#oplot.matrix.hierarchical_cluster_sorted_heatmap)(df[, ...])   | A function to plot a square df (i.e. same indices and columns) that contains distances/similarities as it's values, as a heatmap whose indices and columns are sorted according to a hierarchical clustering (based on the distances listed in the df). |
| `labeled_heatmap`(X[, y, col_labels])                                                             |                                                                                                                                                                                                                                                         |
| [`plot_simil_mat_with_labels`](#oplot.matrix.plot_simil_mat_with_labels)(simil_mat, y[, ...])  | A function that plots similarity matrices, grouping labels together and sorting by descending sum of similarities within a group.                                                                                                                       |
| `vlines_of_matrix`(X[, y, col_labels, ...])                                                       |                                                                                                                                                                                                                                                         |
| [`vlines_ranges`](#oplot.matrix.vlines_ranges)(X[, aggr, axis])                   | vlines plot statistics of X matrix data                                                                                                                                                                                                                 |
| [`xy_boxplot`](#oplot.matrix.xy_boxplot)(X[, y, col_labels, grid_size])        | Boxplot of X (sklearn-like) data, grouped by y (if given)                                                                                                                                                                                               |

### oplot.matrix.get_figsize_to_fit(shape, max_size=11)

Calculate a proportional figsize based on the dimensions of the dataframe, with the larger dimension
capped at `max_size`.

### Parameters

- shape: shape of the matrix (rows, cols).
- max_size: The maximum size for the larger dimension (default 9).

### Returns

- A tuple representing the figsize (width, height).

### oplot.matrix.heatmap(X, y=None, col_labels=None, figsize=None, cmap=None, return_gcf=False, ax=None, xlabel_top=True, ylabel_left=True, xlabel_bottom=True, ylabel_right=True, \*\*kwargs)

Heatmap plot of (X, y) sklearn-like data

### oplot.matrix.heatmap_sns(df, , cmap='Oranges', xlabel=None, ylabel=None, xlabel_fontsize=12, ylabel_fontsize=12, x_tick_fontsize=10, y_tick_fontsize=10, figsize=11, x_tick_rotation=90, y_tick_rotation=0, show_colorbar=False, vert_lines=5, horiz_lines=5, linewidths=0.5, linecolor='white', major_line_color='#D3D3D3', major_line_style='-', vmin=0.2, vmax=1)

A reusable heatmap function to visualize numerical dataframes with customizable options, including adding
vertical and horizontal lines for better readability.

### Parameters

- df: The dataframe to plot.
- cmap: The color map to use (default “Oranges”).
- xlabel: Optional label for the x-axis.
- ylabel: Optional label for the y-axis.
- xlabel_fontsize: Font size for x-axis label (default 12).
- ylabel_fontsize: Font size for y-axis label (default 12).
- tick_fontsize: Font size for tick labels (default 10).
- figsize: Dimensions of the figure (if a tuple) or max dimension size if int (default is 9)
- rotation: Rotation angle for x-tick labels (default 90).
- show_colorbar: Whether to display the colorbar (default False).
- vert_lines: Vertical lines either as an int (step), a list of positions, or None (default 5).
- horiz_lines: Horizontal lines either as an int (step), a list of positions, or None (default 5).
- linewidths: Line width between cells (default 0.5).
- linecolor: Color of the grid lines (default “white”).
- major_line_color: Color of major grid lines (default “#D3D3D3”).
- major_line_style: Style of major grid lines (default “-“).
- vmin: Minimum color value for heatmap (default 0.2 for better contrast).
- vmax: Maximum color value for heatmap (default 1 for better contrast).

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> df = pd.DataFrame(np.random.rand(20, 10), columns=list('ABCDEFGHIJ'))
```

Using default vertical and horizontal lines every 5 rows and columns

```pycon
>>> heatmap_sns(df)
```

Customizing the intervals for vertical and horizontal lines

```pycon
>>> heatmap_sns(df, vert_lines=3, horiz_lines=[4, 8, 12])
```

Specifying exact positions for vertical and horizontal lines

```pycon
>>> heatmap_sns(df, vert_lines=[2, 5, 8], horiz_lines=[1, 6, 11], cmap="Blues", show_colorbar=True)
```

### oplot.matrix.hierarchical_cluster_sorted_heatmap(df, only_return_sorted_df=False, seaborn_heatmap_kwargs=None)

A function to plot a square df (i.e. same indices and columns) that contains distances/similarities as it’s values,
as a heatmap whose indices and columns are sorted according to a hierarchical clustering
(based on the distances listed in the df).

* **Parameters:**
  * **df** – The distance (or similarity) square matrix
  * **only_return_sorted_df** – Default False. Set to True to return the df instead of the heatmap
  * **seaborn_heatmap_kwargs** – the arguments to use in seaborn.heatmap (default is dict(cbar=False))
* **Returns:**
  whatever sns.heatmap returns, or the sorted df if only_return_sorted_df=True

### oplot.matrix.plot_simil_mat_with_labels(simil_mat, y, inner_class_ordering='mean_shift_clusters', brightness=1.0, figsize=(10, 10))

A function that plots similarity matrices, grouping labels together and sorting by descending sum of similarities
within a group.

### oplot.matrix.vlines_ranges(X, aggr=('min', 'median', 'max'), axis=0, \*\*kwargs)

vlines plot statistics of X matrix data

### oplot.matrix.xy_boxplot(X, y=None, col_labels=None, grid_size=None)

Boxplot of X (sklearn-like) data, grouped by y (if given)
