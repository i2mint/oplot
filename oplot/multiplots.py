"""Drawing multiple plots in a single figure"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


def make_space_above(axes, topmargin=1):
    """ increase figure size to make topmargin (in inches) space for
        titles, without changing the axes sizes"""

    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1 - s.top) * h + topmargin
    fig.subplots_adjust(bottom=s.bottom * h / figh, top=1 - topmargin / figh)
    fig.set_figheight(figh)


def ax_func_to_plot(
    list_func_per_ax,
    n_per_row=3,
    title=None,
    title_font_size=10,
    width=15,
    height_row=10,
    saving_path=None,
    x_labels=None,
    y_labels=None,
    outer_axis_labels_only=False,
    dpi=200,
    plot=True,
    h_pad=0,
    w_pad=0,
    title_offset=0,
):
    """
    Draw one grid of plots from the individual plots

    :param list_func_per_ax: a list/generator of functions, each taking an ax object as an input and plotting something on it
    :param n_per_row: number of plots per row
    :param title: global title of the plot
    :param title_font_size: font size of the global title
    :param width: width of the global plot
    :param height_row: height of each row
    :param saving_path: path where to save the plot, can be left to none in which case the plot is not saved
    :param x_labels: label of the x axis
    :param y_labels: label of the y axis
    :param outer_axis_labels_only: if set to true, only the axis labels on the left column and bottom row will show
    :return:
    """

    n_rows = int(np.ceil(len(list_func_per_ax) / n_per_row))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_per_row,
        figsize=(width, height_row * n_rows),
        squeeze=False,
    )

    # fig.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    fig.suptitle(title, fontsize=title_font_size)

    for idx, ax in enumerate(axes.flat):
        if idx < len(list_func_per_ax):
            ax.set(xlabel=x_labels, ylabel=y_labels)

    if outer_axis_labels_only:
        for idx, ax in enumerate(axes.flat):
            if idx < len(list_func_per_ax):
                ax.label_outer()

    for idx, (ax, func) in enumerate(zip(axes.flatten(), list_func_per_ax)):
        if idx < len(list_func_per_ax):
            func(ax=ax)

    # Delete the remaining empty plots if any
    for i in range(len(list_func_per_ax), n_rows * n_per_row):
        fig.delaxes(axes.flatten()[i])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=1)
    plt.tight_layout(h_pad=h_pad, w_pad=w_pad)

    make_space_above(axes, topmargin=title_offset)

    if saving_path:
        fig.savefig(saving_path, dpi=dpi)
    if plot:
        plt.show()


def multiplot_with_max_size(
    list_func_per_ax,
    max_plot_per_file=60,
    n_per_row=3,
    title=None,
    title_font_size=10,
    width=15,
    height_row=10,
    saving_path_format=None,
    x_labels=None,
    y_labels=None,
    outer_axis_labels_only=False,
    dpi=300,
    plot=True,
):
    """
    Same as ax_func_to_plot but saves on several files
    :param max_plot_per_file: the maximum number of plots per file
    """

    n_files, n_remainder_rows = divmod(len(list_func_per_ax), max_plot_per_file)
    file_idx = 0
    for file_idx in range(n_files):
        funcs = list_func_per_ax[
            file_idx * max_plot_per_file : (file_idx + 1) * max_plot_per_file
        ]
        if saving_path_format:
            saving_path = saving_path_format.format(file_idx)
        else:
            saving_path = None
        ax_func_to_plot(
            funcs,
            n_per_row=n_per_row,
            title=title,
            title_font_size=title_font_size,
            width=width,
            height_row=height_row,
            saving_path=saving_path,
            x_labels=x_labels,
            y_labels=y_labels,
            outer_axis_labels_only=outer_axis_labels_only,
        )
    file_idx += 1
    if saving_path_format:
        saving_path = saving_path_format.format(file_idx)
    else:
        saving_path = None
    funcs = list_func_per_ax[-n_remainder_rows:]
    ax_func_to_plot(
        funcs,
        n_per_row=n_per_row,
        title=title,
        title_font_size=title_font_size,
        width=width,
        height_row=height_row,
        saving_path=saving_path,
        x_labels=x_labels,
        y_labels=y_labels,
        outer_axis_labels_only=outer_axis_labels_only,
        dpi=dpi,
        plot=plot,
    )


# # Example of usage
# if __name__ == '__main__':
#     def ax_func(ax):
#         ax.plot([1, 5, 3])
#         ax.set_title('test_test')
#
#
#     ax_func_to_plot([ax_func] * 6, title='Test', x_labels='x_name_here', y_labels='something',
#                     outer_axis_labels_only=True)
