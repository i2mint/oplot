# oplot.distributions

Plot distributions (density etc.) of data.

### Functions

| [`kdeplot_w_boundary_condition`](#oplot.distributions.kdeplot_w_boundary_condition)([data, x, y, ...])   | Custom KDE plot that respects a boundary condition and handles datetime data.   |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|

### oplot.distributions.kdeplot_w_boundary_condition(data=None, , x=None, y=None, boundary_condition=None, ax=None, levels=10, fill=True, cmap=None, figsize=None, \*\*kwargs)

Custom KDE plot that respects a boundary condition and handles datetime data.

This is useful because sometimes when you have data that fits certain conditions
(e.g. y <= x), you want to plot the KDE of the data but only where the condition
is met. If you scatter the data, you can see the boundary, but the KDE plot will
not respect the boundary. This function allows you to specify a boundary condition
and only plot the KDE where the condition is met.

### Parameters

- data: DataFrame, optional
  : Dataset for plotting.
- x, y: vectors or keys in `data`
  : Variables that specify positions on the x and y axes.
- boundary_condition: function
  : Function that takes arrays of x and y values and returns a boolean array
    indicating where the density should be zero.
- ax: matplotlib Axes, optional
  : Axes object to draw the plot onto; otherwise, uses the current Axes.
- \*\*kwargs: dict
  : Additional keyword arguments passed to matplotlib contour functions.

### Returns

- ax: matplotlib Axes
  : The Axes object with the plot drawn onto it.

### Examples

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
```

Generate sample data

```pycon
>>> np.random.seed(42)
>>> x = np.random.normal(0, 1, 500)
>>> y = np.random.normal(0, 1, 500)
>>> data = pd.DataFrame({'x': x, 'y': y})
```

Define a boundary condition

```pycon
>>> boundary_condition = lambda X, Y: Y <= X
```

Plot using the custom KDE function

```pycon
>>> ax = kdeplot_w_boundary_condition(
...     data=data,
...     x='x',
...     y='y',
...     boundary_condition=boundary_condition,
...     fill=True,
...     cmap='viridis',
...     levels=15
... )
```
