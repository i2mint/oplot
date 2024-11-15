"""Plot distributions (density etc.) of data."""

import datetime

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.dates import date2num, num2date

from matplotlib import dates as mdates


def kdeplot_w_boundary_condition(
    data=None,
    *,
    x=None,
    y=None,
    boundary_condition=None,
    ax=None,
    levels=10,
    fill=True,
    cmap=None,
    figsize=None,
    **kwargs
):
    """
    Custom KDE plot that respects a boundary condition and handles datetime data.

    This is useful because sometimes when you have data that fits certain conditions 
    (e.g. y <= x), you want to plot the KDE of the data but only where the condition 
    is met. If you scatter the data, you can see the boundary, but the KDE plot will
    not respect the boundary. This function allows you to specify a boundary condition
    and only plot the KDE where the condition is met. 

    

    Parameters:
    - data: DataFrame, optional
        Dataset for plotting.
    - x, y: vectors or keys in `data`
        Variables that specify positions on the x and y axes.
    - boundary_condition: function
        Function that takes arrays of x and y values and returns a boolean array
        indicating where the density should be zero.
    - ax: matplotlib Axes, optional
        Axes object to draw the plot onto; otherwise, uses the current Axes.
    - **kwargs: dict
        Additional keyword arguments passed to matplotlib contour functions.

    Returns:
    - ax: matplotlib Axes
        The Axes object with the plot drawn onto it.


    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    
    Generate sample data

    >>> np.random.seed(42)
    >>> x = np.random.normal(0, 1, 500)
    >>> y = np.random.normal(0, 1, 500)
    >>> data = pd.DataFrame({'x': x, 'y': y})
    
    Define a boundary condition
    
    >>> boundary_condition = lambda X, Y: Y <= X
    
    Plot using the custom KDE function
    
    >>> ax = kdeplot_w_boundary_condition(
    ...     data=data,
    ...     x='x',
    ...     y='y',
    ...     boundary_condition=boundary_condition,
    ...     fill=True,
    ...     cmap='viridis',
    ...     levels=15
    ... )

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        assert figsize is None, "figsize should not be provided if ax is provided."

    if boundary_condition is None:
        return sns.kdeplot(
            data=data, x=x, y=y, ax=ax, levels=levels, fill=fill, cmap=cmap, **kwargs
        )

    # Handle data and extract x and y
    if data is not None:
        if isinstance(x, str):
            x = data[x]
        if isinstance(y, str):
            y = data[y]
    if x is None or y is None:
        raise ValueError("Both x and y must be provided.")

    x = np.asarray(x)
    y = np.asarray(y)

    # Process datetime data
    x_is_datetime = np.issubdtype(x.dtype, np.datetime64) or isinstance(
        x[0], datetime.datetime
    )
    y_is_datetime = np.issubdtype(y.dtype, np.datetime64) or isinstance(
        y[0], datetime.datetime
    )

    x_values = date2num(x) if x_is_datetime else x
    y_values = date2num(y) if y_is_datetime else y

    # Perform KDE estimation
    values = np.vstack([x_values, y_values])
    kde = gaussian_kde(values)

    # Create grid
    xmin, xmax = x_values.min(), x_values.max()
    ymin, ymax = y_values.min(), y_values.max()

    num_grid_points = kwargs.pop('gridsize', 100)
    x_grid = np.linspace(xmin, xmax, num_grid_points)
    y_grid = np.linspace(ymin, ymax, num_grid_points)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Evaluate the density
    Z = np.reshape(kde(positions).T, X.shape)

    # Apply boundary condition
    if boundary_condition is not None:
        # boundary_condition should be a function that takes arrays of x and y and returns a boolean array
        mask = boundary_condition(X, Y)
        Z = np.where(mask, Z, 0)
    else:
        raise ValueError("A boundary_condition function must be provided.")

    # Plot the density
    if ax is None:
        ax = plt.gca()

    if fill:
        contour_func = ax.contourf
    else:
        contour_func = ax.contour

    contour_func(X, Y, Z, levels=levels, cmap=cmap)

    # Format axes for datetime if necessary
    if x_is_datetime:
        ax.set_xlim(date2num(num2date(X.min())), date2num(num2date(X.max())))
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )
        plt.setp(ax.get_xticklabels(), rotation=45)
    if y_is_datetime:
        ax.set_ylim(date2num(num2date(Y.min())), date2num(num2date(Y.max())))
        ax.yaxis_date()
        ax.yaxis.set_major_locator(mdates.AutoDateLocator())
        ax.yaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.yaxis.get_major_locator())
        )
        plt.setp(ax.get_yticklabels(), rotation=45)

    # Set labels
    ax.set_xlabel(kwargs.get('xlabel', 'x'))
    ax.set_ylabel(kwargs.get('ylabel', 'y'))

    return ax

