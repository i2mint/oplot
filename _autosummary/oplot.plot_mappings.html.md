# oplot.plot_mappings

Plot mappings (e.g. dicts, Series, etc.)

### Functions

| [`dict_bar_plot`](#oplot.plot_mappings.dict_bar_plot)(d[, title, figsize, ...])   | Plot a bar plot from a dictionary.   |
|--------------------------------------------------------------------------------------------|--------------------------------------|

### oplot.plot_mappings.dict_bar_plot(d, title='', figsize=(12, 5), , numeric_x_axis=None, xlabel=None, ylabel=None, annotations=None, annotations_cutoff_length=None, annotations_font_size=None, annotations_font_size_width_factor=12, annotations_rotation=90)

Plot a bar plot from a dictionary.

Parameters

d
: Dictionary or Series to plot.

title
: Title of the plot.

figsize
: Size of the plot.

numeric_x_axis
: If True, x-axis is treated as numeric.

xlabel
: Label of the x-axis.

ylabel
: Label of the y-axis.

annotations
: Dictionary of annotations to add to the plot.

annotations_cutoff_length
: Maximum length of annotations.

annotations_font_size
: Font size of annotations.

annotations_font_size_width_factor
: Factor to adjust font size based on bar width.

annotations_rotation
: Rotation of annotations.
