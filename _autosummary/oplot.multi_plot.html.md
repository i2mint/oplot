# oplot.multi_plot

Plotting multiple datas in a same figure

### Functions

| [`ax_func_to_plot`](#oplot.multi_plot.ax_func_to_plot)(list_func_per_ax[, ...])    | Each function in list_func_per_ax takes an ax as input and draw something on it   |
|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`multi_row_plot`](#oplot.multi_plot.multi_row_plot)([data_list, plot_func, ...]) | Quickly plotting multiple rows of data.                                           |

### oplot.multi_plot.ax_func_to_plot(list_func_per_ax, n_per_row=3, title=None, title_font_size=10, width=15, height_row=10, saving_path=None, rec_padding=(0, 0, 0, 0), x_labels=None, y_labels=None, outer_axis_labels_only=False, show=True)

Each function in list_func_per_ax takes an ax as input and draw something on it

outer_axis_labels_only: if set to true, only the axis labels on the left column and bottom row will show
x_labels: the label on all x-axis
y_labels: the label on all the y-axis

### oplot.multi_plot.multi_row_plot(data_list=(), plot_func=<function plot>, figsize=3, plot_func_kwargs=None, ax_calls=())

Quickly plotting multiple rows of data.

* **Parameters:**
  * **data_list** – 

    The list of datas to plot. For each “row_data” of data_list, a row will be created and plot_func
    will be called, using that item as input. If row_data is:
    > * a dict, plot_func(\*\*dict(plot_func_kwargs, \*\*row_data)) will be called to populate that row
    > * a tuple, plot_func(\*row_data, \*\*plot_func_kwargs) will be called to populate that row
    > * if not, plot_func(row_data, \*\*plot_func_kwargs) will be called to populate that row
  * **plot_func** – The plotting function to use.
  * **figsize** – 

    The figsize to use. If
    * a tuple of length 2, figure(figsize=figsize) will be called to create the figure
    * a number (int or float), figure(figsize=(16, n_rows \* figsize_units_per_row)) will be called
    * If None, figure won’t be called (we assume therefore, it’s been created already, for example
  * **plot_func_kwargs** – The kwargs to use as arguments of plot_func for every data row.
  * **ax_calls** – 

    A list of (attr, args, kwargs) triples that will result in calling
    : getattr(ax, attr)(\*args, \*\*kwargs)

    for every ax in ax_list (the list of row axes)
* **Returns:**
  ax_list, the list of axes for each row
