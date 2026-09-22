# oplot.plot_data_set

Function to reduce and plot data in 2 or 3 dimensions

Example of usage:

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, n_features=4, centers=3, cluster_std=1.0)

y_conf = []
for i in range(len(y)):
…if y[i] == 0 or y[i] == 2:
…    y_conf.append(0)
…else:
…    y_conf.append(1)
y_conf = np.array(y_conf)

scatter_and_color_according_to_y(X, y_conf, col=’rainbow’, dim_reduct=’LDA’, projection=’3d’)

### Functions

| [`density_distribution`](#oplot.plot_data_set.density_distribution)(data_dict, \*[, ax, ...])   | Plots the density distribution of different data sets (arrays).                                   |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `ensure_dict`(obj)                                                                                |                                                                                                   |
| [`ratio_comparison_vlines`](#oplot.plot_data_set.ratio_comparison_vlines)(y1, y2[, c1, c2])        | Plots vlines of y1/y2.                                                                            |
| [`save_figs_to_pdf`](#oplot.plot_data_set.save_figs_to_pdf)(figs[, pdf_filepath])           | Save figures to a single pdf                                                                      |
| [`scatter_and_color_according_to_y`](#oplot.plot_data_set.scatter_and_color_according_to_y)(X[, y, ...])    |                                                                                                   |
| [`side_by_side_bar`](#oplot.plot_data_set.side_by_side_bar)(list_of_values_for_bars[, ...]) | A plotting utility making side by side bar graphs from a list of list (of same length) of values. |

### oplot.plot_data_set.density_distribution(data_dict, \*, ax=None, axvline_kwargs=None, line_width=3, location_func=<function mean>, location_linestyle='--', display_location_text=True, colors=('blue', 'orange', 'green', 'red', 'purple', 'brown'), density_plot_func=<function kdeplot>, density_plot_kwargs=None, text_kwargs=(('x', 0.05), ('y', 0.05), ('bbox', {'alpha': 0.5, 'facecolor': 'white'})), mean_line_kwargs=None)

Plots the density distribution of different data sets (arrays).

* **Parameters:**
  * **data_dict** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), `ndarray`]) – A dictionary where keys are labels and values are arrays to plot.
  * **ax** (`Axes` | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Matplotlib Axes object to plot on. If None, a new figure and axis will be created.
  * **axvline_kwargs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A dictionary where keys are labels and values
    are dictionaries of axvline kwargs.
    If not provided, default colors and linestyle will be used.
  * **line_width** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Width of the density plot lines.
  * **display_means** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool) *,* *optional*) – Whether to display the means as text on the plot.
  * **means_linestyle** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *,* *optional*) – Linestyle for the mean vertical lines.
  * **colors** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – Tuple of colors to cycle through for the plots.
  * **density_plot_func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – Function to use for density plotting.
  * **density_plot_kwargs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Additional keyword arguments for density_plot_func.
  * **text_kwargs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Additional keyword arguments for plt.text.
  * **mean_line_kwargs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Additional keyword arguments for plt.axvline.

### Example

```pycon
>>> import numpy as np
>>> data_dict = {
...     'dist1': np.random.normal(0, 1, 100),
...     'dist2': np.random.normal(5, 2, 100)
... }
>>> density_distribution(data_dict)
>>> # This will plot the density distributions of dist1 and dist2 with vertical lines at their means.
```

```pycon
>>> fig, ax = plt.subplots()
>>> density_distribution(data_dict, ax=ax, display_location_text=False, colors=('red', 'blue'))
>>> # This will plot the density distributions on the provided axis.
```

### oplot.plot_data_set.ratio_comparison_vlines(y1, y2, c1='b', c2='k')

Plots vlines of y1/y2.

* **Parameters:**
  * **y1** – numerator
  * **y2** – denominator
  * **c1** – color of numerator
  * **c2** – color of denominator (will be a straight horizontal line placed at 1)
* **Returns:**
  what plt.plot returns

### oplot.plot_data_set.save_figs_to_pdf(figs, pdf_filepath=None)

Save figures to a single pdf

* **Parameters:**
  * **figs**
  * **pdf_filepath**
* **Returns:**

### oplot.plot_data_set.scatter_and_color_according_to_y(X, y=None, col='rainbow', projection='2d', dim_reduct='LDA', save=False, legend=True, saving_loc='/home/chris/', saving_name='myplot-', plot_tag_name=False, super_alpha=10, cmap_col='viridis', \*args, \*\*kwargs)

* **Parameters:**
  * **X** – an array of feature vectors
  * **y** – an array of tags
  * **col** – ‘random’ or ‘rainbow’. Rainbow tends to give more distinct colors
  * **projection** – ‘2d’ or ‘3d’
  * **dim_reduct** – ‘LDA’ or ‘PCA’, the dimension reduction method used (when needed)
    anything else and the function will use the first 2 or 3 coordinates by default
  * **iterated** – 

    whether or not to use iterated projection in LDA when the number of tags is 2
    : without the iterate projection, one would get only 1 dimension out of lda
      if set to False and the number of tags is 2 and the original space has dimension
      more than 2, the first two dimensions will be retained for the scatterplot
      (i.e. no smart drawing then). This last option can be useful when the iterated
      projection just yield points on a line.

    fall_back: when LDA cannot produce the required number of dimensions, PCA is used instead.
  * **save** – a boolean, whether or not the plot will be saved
  * **saving_loc** – a string, the location where the file will be saved. If none is given,
    it will be saved in the home folder
  * **args** – more argument for scatter
  * **kwargs** – more keyword argument for scatter
* **Returns:**
  a plot of 2d scatter plot of X with different colors for each tag

### oplot.plot_data_set.side_by_side_bar(list_of_values_for_bars, width=1, spacing=1, list_names=None, colors=None)

A plotting utility making side by side bar graphs from a list of list (of same length) of values.

* **Parameters:**
  * **list_of_values_for_bars** – list of list of values for the bar graphs
  * **width** – the width of the bar graphs
  * **spacing** – the size of the spacing between groups of bars
  * **list_names** – the names to assign to the bars, in the same order as in list_of_values_for_bars
  * **list_colors** – the colors to use for the bars, in the same order as in list_of_values_for_bars
* **Returns:**
  a nice plot!
