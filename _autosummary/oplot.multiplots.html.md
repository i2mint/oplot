# oplot.multiplots

Drawing multiple plots in a single figure

### Functions

| [`ax_func_to_plot`](#oplot.multiplots.ax_func_to_plot)(list_func_per_ax[, ...])         | Draw one grid of plots from the individual plots                                                     |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| [`make_space_above`](#oplot.multiplots.make_space_above)(axes[, topmargin])              | increase figure size to make topmargin (in inches) space for titles, without changing the axes sizes |
| [`multiplot_with_max_size`](#oplot.multiplots.multiplot_with_max_size)(list_func_per_ax[, ...]) | Same as ax_func_to_plot but saves on several files                                                   |

### oplot.multiplots.ax_func_to_plot(list_func_per_ax, n_per_row=3, title=None, title_font_size=10, width=15, height_row=10, saving_path=None, x_labels=None, y_labels=None, outer_axis_labels_only=False, dpi=200, plot=True, h_pad=0, w_pad=0, title_offset=0)

Draw one grid of plots from the individual plots

* **Parameters:**
  * **list_func_per_ax** – a list/generator of functions, each taking an ax object as an input and plotting something on it
  * **n_per_row** – number of plots per row
  * **title** – global title of the plot
  * **title_font_size** – font size of the global title
  * **width** – width of the global plot
  * **height_row** – height of each row
  * **saving_path** – path where to save the plot, can be left to none in which case the plot is not saved
  * **x_labels** – label of the x axis
  * **y_labels** – label of the y axis
  * **outer_axis_labels_only** – if set to true, only the axis labels on the left column and bottom row will show
* **Returns:**

### oplot.multiplots.make_space_above(axes, topmargin=1)

increase figure size to make topmargin (in inches) space for
titles, without changing the axes sizes

### oplot.multiplots.multiplot_with_max_size(list_func_per_ax, max_plot_per_file=60, n_per_row=3, title=None, title_font_size=10, width=15, height_row=10, saving_path_format=None, x_labels=None, y_labels=None, outer_axis_labels_only=False, dpi=300, plot=True)

Same as ax_func_to_plot but saves on several files

* **Parameters:**
  **max_plot_per_file** – the maximum number of plots per file
