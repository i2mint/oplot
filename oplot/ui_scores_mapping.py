"""Functions to create and plot outlier scores (or other) in a fixed bounded range. Intended to use to
show the results of an outlier algorithm in a user friendly UI"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from omodel.outliers.ui_score_function import *


def make_slidable_sigmoid(min_score=0, max_score=5,
                          plot_min=None, plot_max=None,
                          gran=.01):
    """
    usage in a notebook:

    %matplotlib notebook
    from sound_sketch.ca.math_utils.sigmoid_type_functions import *
    make_slidable_sigmoid()
    """

    if not plot_min:
        plot_min = min_score - (max_score - min_score)
    if not plot_max:
        plot_max = max_score + (max_score - min_score)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(plot_min, plot_max, gran)

    top = 2
    bottom = 2
    base = make_ui_score_mapping(top_base=top, bottom_base=bottom,
                                 max_lin_score=max_score, min_lin_score=min_score)
    s = base(t)
    l, = plt.plot(t, s, lw=2)
    ax.margins(x=0)

    axcolor = 'lightgoldenrodyellow'
    axbottom = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axtop = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    sbottom = Slider(axbottom, 'bottom', 1, 5.0, valinit=top)
    stop = Slider(axtop, 'top', 1, 5.0, valinit=bottom)

    def update(val):
        """
        Util function to help updating the plot
        :param val:
        :return:
        """
        top = stop.val
        bottom = sbottom.val
        f = make_ui_score_mapping(top_base=top, bottom_base=bottom,
                                  max_lin_score=max_score, min_lin_score=min_score)
        l.set_ydata(f(t))
        fig.canvas.draw_idle()

    sbottom.on_changed(update)
    stop.on_changed(update)
    plt.show()

