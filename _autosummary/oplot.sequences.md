# oplot.sequences

Plotting sequences

### Functions

| [`bars`](#oplot.sequences.bars)(data[, y, x, hue, density_line, ...])   | Create a customizable barplot with optional zero replacement and an overlayed density (smoothed) line.   |
|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| [`fill_zeros`](#oplot.sequences.fill_zeros)(array, fill_val)                  | Replace zeros in the input array with the specified fill value.                                          |

### oplot.sequences.bars(data, y=None, , x=None, hue=None, density_line=False, figsize=(18, 5), dodge=False, width=1, x_ticks=None, title=None, x_label=None, y_label=None, zero_thickness=None, barplot_kwargs=(), density_sigma=20, density_line_kwargs=(('color', 'black'),), ax=None)

Create a customizable barplot with optional zero replacement and an overlayed density (smoothed) line.

This function generates a seaborn barplot using the provided y values and an optional DataFrame.
The y values can be given directly as an array or specified by a column name (or index) in ‘data’.
If zero_thickness is enabled, any zero values in the y data are replaced with a small value
(computed as a fraction of the y-range) so that they are visibly rendered when a hue grouping or x values
are provided. The x values are taken from the column specified by ‘x’ (if provided), or default to a range
matching the length of y. Optionally, a smoothed density line is overlayed by applying a Gaussian filter
(with sigma controlled by density_sigma) to the y data, with additional styling provided via density_line_kwargs.

* **Parameters:**
  * **data** (*pandas.DataFrame* *or* *None*) – DataFrame containing the data for plotting or the y array to be plotted itself (in which case, the y argument should be None).
  * **y** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *, or* *array-like*) – Either a column name (or index) to select the y values from ‘data’, or an array-like of y-axis values.
  * **x** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *, or* *None* *,* *optional*) – Column name (or index) to use for x-axis values from ‘data’. If None, x values default to range(len(y)).
  * **hue** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *or* *None* *,* *optional*) – Column name in ‘data’ used for hue grouping (default is None).
  * **density_line** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool) *,* *optional*) – If True, overlays a smoothed density line on the barplot computed from the y values
    (default is False).
  * **figsize** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Size of the figure as (width, height) (default is (18, 5)).
  * **dodge** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool) *,* *optional*) – Whether to dodge the bars when grouping by hue (default is False).
  * **width** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Width of the bars (default is 1).
  * **x_ticks** (*array-like* *or* *None* *,* *optional*) – Custom positions for x-axis ticks. If None and x is not specified, ticks are removed.
  * **title** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *or* *None* *,* *optional*) – Title of the plot (default is None).
  * **x_label** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *or* *None* *,* *optional*) – Label for the x-axis (default is None).
  * **y_label** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *or* *None* *,* *optional*) – Label for the y-axis (default is None).
  * **zero_thickness** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Value used to replace zero values in the y data. If None, a replacement is activated
    when either x or hue is provided. If set to True, a default factor of 0.02 is used; if 0 or False,
    zero replacement is disabled.
  * **barplot_kwargs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Additional keyword arguments to pass to seaborn.barplot.
  * **density_sigma** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Sigma parameter for the Gaussian filter used to smooth the y values for the density line
    (default is 20).
  * **density_line_kwargs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Additional keyword arguments for styling the density line (default is { ‘color’: ‘black’ }).
  * **ax** (*matplotlib.axes.Axes* *or* *None* *,* *optional*) – Axes object to draw the plot onto, otherwise creates a new figure and axes (default is None).
* **Returns:**
  The Axes object containing the final plot.
* **Return type:**
  matplotlib.axes.Axes

### oplot.sequences.fill_zeros(array, fill_val)

Replace zeros in the input array with the specified fill value.

* **Parameters:**
  * **array** (`ndarray`) – Input array.
  * **fill_val** (*scalar*) – Value to replace zeros with.
* **Returns:**
  Array with zeros replaced by fill_val.
* **Return type:**
  np.ndarray
