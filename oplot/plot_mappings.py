"""Plot mappings (e.g. dicts, Series, etc.)"""


def dict_bar_plot(
    d: dict,
    title='',
    figsize=(12, 5),
    *,
    numeric_x_axis=None,
    xlabel=None,
    ylabel=None,
    annotations=None,
    annotations_cutoff_length=None,
    annotations_font_size=None,
    annotations_font_size_width_factor=12,
    annotations_rotation=90,
):
    """
    Plot a bar plot from a dictionary.

    Parameters

    d : dict or pd.Series
        Dictionary or Series to plot.
    title : str
        Title of the plot.
    figsize : tuple
        Size of the plot.
    numeric_x_axis : bool
        If True, x-axis is treated as numeric.
    xlabel : str
        Label of the x-axis.
    ylabel : str
        Label of the y-axis.
    annotations : dict
        Dictionary of annotations to add to the plot.
    annotations_cutoff_length : int
        Maximum length of annotations.
    annotations_font_size : int
        Font size of annotations.
    annotations_font_size_width_factor : float
        Factor to adjust font size based on bar width.
    annotations_rotation : int
        Rotation of annotations.


    """
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(d, pd.Series):
        d = d.to_dict()
        xlabel = xlabel or d.index.name
        ylabel = ylabel or d.name

    x = list(d.keys())
    y = list(d.values())

    # Determine if x-axis should be numeric
    if numeric_x_axis is None:
        try:
            x_numeric = np.array(x, dtype=float)
            numeric_x_axis = True
        except ValueError:
            numeric_x_axis = False

    if numeric_x_axis:
        x_numeric = np.array([float(k) for k in x])
        # Sort x and y according to x
        sorted_indices = np.argsort(x_numeric)
        x_sorted = x_numeric[sorted_indices]
        y_sorted = np.array(y)[sorted_indices]
        keys_sorted = np.array(x)[sorted_indices]  # Original keys

        # Compute bar width
        dx = np.diff(x_sorted)
        if len(dx) > 0:
            min_dx = np.min(dx)
        else:
            min_dx = 1
        bar_width = min_dx * 0.8

        plt.figure(figsize=figsize)
        bars = plt.bar(
            x_sorted,
            y_sorted,
            width=bar_width,
            align='center',
            edgecolor='black',
            color='skyblue',
        )

        # Map original keys to bars
        key_to_bar = {key: bar for key, bar in zip(keys_sorted, bars)}
        y_max = max(y_sorted)
    else:
        # Use seaborn.barplot for non-numerical keys
        sns.set_style('whitegrid')
        sns.set_context('talk')
        sns.set_palette('muted')
        plt.figure(figsize=figsize)
        ax = sns.barplot(x=x, y=y)
        sns.despine()
        bars = ax.patches

        # Map original keys to bars
        key_to_bar = {key: bar for key, bar in zip(x, bars)}
        y_max = max(y)

    plt.title(title)
    plt.grid(axis='y', linestyle='dotted', color='black')
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    # Add annotations if provided
    if annotations:
        if annotations_cutoff_length:

            def new_annotations():
                for k, v in annotations.items():
                    if len(v) > annotations_cutoff_length:
                        yield k, v[:annotations_cutoff_length] + '...'
                    else:
                        yield k, v

            annotations = dict(new_annotations())

        y_offset = y_max * 0.01  # Small offset above the bar
        for key, text in annotations.items():
            if key in key_to_bar:
                bar = key_to_bar[key]
                x_pos = bar.get_x() + bar.get_width() / 2
                y_pos = bar.get_height() + y_offset
                if annotations_font_size is None:
                    # Adjust font size based on bar width
                    annotations_font_size = max(
                        8, bar.get_width() * annotations_font_size_width_factor
                    )

                plt.text(
                    x_pos,
                    y_pos,
                    text,
                    ha='center',
                    va='bottom',
                    fontsize=annotations_font_size,
                    rotation=annotations_rotation,
                )
            else:
                print(f"Warning: Key '{key}' in annotations not found in d dict.")

    plt.show()
