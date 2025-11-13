"""Plotting sequences"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


def bars(
    data,
    y=None,
    *,
    x=None,
    hue=None,
    density_line=False,
    figsize: tuple[float, float] = (18, 5),
    dodge=False,
    width: int = 1,
    x_ticks=None,
    title=None,
    x_label=None,
    y_label=None,
    zero_thickness: float | None = None,
    barplot_kwargs: dict = (),
    density_sigma: int = 20,
    density_line_kwargs: dict = (('color', 'black'),),
    ax=None,  # Add an ax parameter
):
    """
    Create a customizable barplot with optional zero replacement and an overlayed density (smoothed) line.

    This function generates a seaborn barplot using the provided y values and an optional DataFrame.
    The y values can be given directly as an array or specified by a column name (or index) in 'data'.
    If zero_thickness is enabled, any zero values in the y data are replaced with a small value
    (computed as a fraction of the y-range) so that they are visibly rendered when a hue grouping or x values
    are provided. The x values are taken from the column specified by 'x' (if provided), or default to a range
    matching the length of y. Optionally, a smoothed density line is overlayed by applying a Gaussian filter
    (with sigma controlled by density_sigma) to the y data, with additional styling provided via density_line_kwargs.

    Parameters
    ----------
    data : pandas.DataFrame or None
        DataFrame containing the data for plotting or the y array to be plotted itself (in which case, the y argument should be None).
    y : str, int, or array-like
        Either a column name (or index) to select the y values from 'data', or an array-like of y-axis values.
    x : str, int, or None, optional
        Column name (or index) to use for x-axis values from 'data'. If None, x values default to range(len(y)).
    hue : str or None, optional
        Column name in 'data' used for hue grouping (default is None).
    density_line : bool, optional
        If True, overlays a smoothed density line on the barplot computed from the y values
        (default is False).
    figsize : tuple of float, optional
        Size of the figure as (width, height) (default is (18, 5)).
    dodge : bool, optional
        Whether to dodge the bars when grouping by hue (default is False).
    width : int, optional
        Width of the bars (default is 1).
    x_ticks : array-like or None, optional
        Custom positions for x-axis ticks. If None and x is not specified, ticks are removed.
    title : str or None, optional
        Title of the plot (default is None).
    x_label : str or None, optional
        Label for the x-axis (default is None).
    y_label : str or None, optional
        Label for the y-axis (default is None).
    zero_thickness : float or None, optional
        Value used to replace zero values in the y data. If None, a replacement is activated
        when either x or hue is provided. If set to True, a default factor of 0.02 is used; if 0 or False,
        zero replacement is disabled.
    barplot_kwargs : dict, optional
        Additional keyword arguments to pass to seaborn.barplot.
    density_sigma : int, optional
        Sigma parameter for the Gaussian filter used to smooth the y values for the density line
        (default is 20).
    density_line_kwargs : dict, optional
        Additional keyword arguments for styling the density line (default is { 'color': 'black' }).
    ax : matplotlib.axes.Axes or None, optional
        Axes object to draw the plot onto, otherwise creates a new figure and axes (default is None).

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the final plot.
    """
    if y is None:
        y = data
        data = _data_frame_not_given

    if isinstance(y, (str, int)):
        y_by = y
        y = data[y_by]

    if density_line:  # need to compute this now, before changing y
        y_density = gaussian_filter1d(y, sigma=density_sigma)

    # Determine zero replacement: if zero_thickness is None, enable it if x or hue is provided.
    if zero_thickness is None:
        zero_thickness = (x is not None) or (hue is not None)
    if zero_thickness:
        if zero_thickness is True:
            zero_thickness = 0.02
        zero_absolute_thickness = zero_thickness * (y.max() - y.min())
        y = fill_zeros(y, zero_absolute_thickness)

    if isinstance(x, (str, int)):
        # Use the provided x column from data.
        x_by = x
        x = data[x_by]
    else:
        # Default x values as a range if no x is provided.
        x = range(len(y))
        if not x_ticks:
            x_ticks = []

    # If the sentinel value was used, reset data to None.
    if data is _data_frame_not_given:
        data = None

    # Replace the implicit creation of axes by using the provided ax or creating a new one
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Use the provided ax for plotting
    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        dodge=dodge,
        width=width,
        ax=ax,
        **dict(barplot_kwargs),
    )

    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    # Don't set figure size if ax was provided externally
    if figsize is not None and ax is None:
        ax.figure.set_size_inches(*figsize)
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if density_line:
        y_density = gaussian_filter1d(y, sigma=density_sigma)
        if 'linewidth' in density_line_kwargs:
            density_line_kwargs['linewidth'] = int(density_line_kwargs['linewidth'])

        sns.lineplot(
            y=y_density,
            x=range(len(y)),
            ax=ax,
            **dict(density_line_kwargs),
        )

    return ax


# --------------------------------------------------------------------------------------
# Utils


def fill_zeros(array: np.ndarray, fill_val):
    """
    Replace zeros in the input array with the specified fill value.

    Parameters
    ----------
    array : np.ndarray
        Input array.
    fill_val : scalar
        Value to replace zeros with.

    Returns
    -------
    np.ndarray
        Array with zeros replaced by fill_val.
    """
    array = np.asarray(array)
    return np.where(array == 0, fill_val, array)


class __data_frame_not_given:
    def __getitem__(self, item):
        raise ValueError("DataFrame not provided")


_data_frame_not_given = __data_frame_not_given()
