"""Functions to create and plot outlier scores (or other) in a fixed bounded range. Intended to use to
show the results of an outlier algorithm in a user friendly UI"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


# def double_sigmoid(joint, min_lower, max_lower, max_upper, base_lower=2, base_upper=2):
#     """
#
#     :param joint: float, the x-axis value where the two sigmoids meet
#     :param min_lower: float, the smallest value taken by the lower sigmoid (as a limit when x -> -inf)
#     :param max_lower: float, the largest value taken by the lower sigmoid (as a limit when x -> inf, note that
#                       the lower sigmoid is cut off at joint and so it will have a maximum strictly less than max_lower)
#     :param max_upper: float, the largest value of the sigmoid (as a limit when x -> inf)
#     :return: a function
#     """
#
#     lower = lambda x: min_lower + (max_lower - min_lower) / (1 + base_lower ** -x)
#     upper = lambda x: lower(joint) - (max_upper - max_lower) / (1 + base_upper ** joint) + (max_upper - max_lower) \
#                       / (1 + base_upper ** -(x - 2 * joint))
#
#     def piecewise(x):
#         if x <= joint:
#             return lower(x)
#         else:
#             return upper(x)
#
#     return piecewise
#
#
# def make_smooching_scores_map(min_lin_range, max_lin_range,
#                               prop_under, prop_above):
#     """
#     Makes a function mapping (-inf, inf) to (0, 1).
#     The function is linear equal to y=x on the range (min_lin_range, max_lin_range), exponential before min_lin_range
#     and also after max_lin_range. The function is smooth and resembles a sigmoid but the linear portion is controlled
#     as well as the relative size of what is under/above the linear portion.
#     This function was made with the UI needs in mind.
#
#     :param min_lin_range: the smallest value of range on which the function is linear
#     :param max_lin_range: the largest value of range on which the function is linear
#     :param prop_under: the proportion of the total y_axis range taken by the curve for all values above max_lin_range
#     :param prop_above: the proportion of the total y_axis range taken by the curve for all values under min_lin_range
#     :return: a sigmoid like function
#     """
#
#     linear_span = max_lin_range - min_lin_range
#     under_span = linear_span * prop_under / (1 - prop_under - prop_above)
#     above_span = linear_span * prop_above / (1 - prop_under - prop_above)
#     under_power = np.exp(1 / under_span)
#     above_power = np.exp(1 / above_span)
#     normalization = max_lin_range + above_span
#
#     def mapping(x):
#         if x < min_lin_range:
#             return ((min_lin_range - under_span) + under_span * under_power ** (x - min_lin_range)) / normalization
#         elif x > max_lin_range:
#             return (max_lin_range + above_span - above_span * above_power ** (max_lin_range - x)) / normalization
#         else:
#             return x / normalization
#
#     return np.vectorize(mapping)


# TODO: add a multiplicative factor in the top model to give the ability to match the top size
# same at the bottom
# probably can't use exponentials but rather functions of the form a / (x-b) + c
def make_linear_part(max_score, min_score):
    """
    :param bottom: the proportion of the graph used for the bottom "sigmoid"
    :param middle: the proportion of the graph used for the middle linear part
    :param top: the proportion of the graph used for the top "sigmoid"
    :param max_score: the maximum score seen on train
    :param min_score: the minimum score seen on train
    :return: the linear part of the ui score mapping
    """

    slope = 1 / (max_score - min_score)

    def linear_part(x):
        return x * slope + 1 - slope * min_score

    return linear_part


def make_top_part(base, max_score, min_score):
    """
    The base has to be between 0 and 1, strictly.
    The function will be of the form -base ** (-x + t) + C, where t and C
    are the two constants to solve for. The constraints are continuity and
    smoothness at max_score when pieced with the linear part
    """

    slope = 1 / (max_score - min_score)
    t = np.log(slope / np.log(base)) / np.log(base) + max_score
    # at the limit when x->inf, the function will approach c
    c = 2 + base ** (-max_score + t)

    def top_part(x):
        return - base ** (-x + t) + c

    return top_part, c


def make_bottom_part(base, max_score, min_score):
    """
    The base has to be between 0 and 1, strictly.
    The function will be of the form -base ** (-x + t) + C, where t and C
    are the two constants to solve for. The constraints are continuity and
    smoothness at max_score when pieced with the linear part
    """

    slope = 1 / (max_score - min_score)
    t = np.log(slope / np.log(base)) / np.log(base) - min_score
    # at the limit when x->-inf, the function will approach c
    c = 1 - base ** (min_score + t)

    def bottom_part(x):
        return base ** (x + t) + c

    return bottom_part, c


def make_ui_score_mapping(min_lin_score, max_lin_score,
                          top_base=2, bottom_base=2,
                          max_score=10,
                          reverse=False):
    """
    Plot a sigmoid function to map outlier scores to (by default) the range (0, 10)
    The function is not only continuous but also smooth and the radius of the corners are controlled by the floats
    top_base and bottom_base
    :param min_lin_score: float, the minimum scores which is map with a linear function
    :param max_lin_score: float, the maximum scores which is map with a linear function
    :param top_base: float, the base of the exponential function on top of the linear part
    :param bottom_base:  float, the base of the exponential function on the bottom of the linear part
    :param max_score: float, the upper bound of the function
    :param reverse: boolean, whether to mirror the function along its center
    :return: a mapping, sigmoid like


    ------------------------ Example of use: ---------------------------

    from oplot.ui_scores_mapping import make_ui_score_mapping
    import numpy as np
    import matplotlib,pyplot as plt

    sigmoid_map = make_ui_score_mapping(min_lin_score=1,
                                        max_lin_score=9,
                                        top_base=2,
                                        bottom_base=2,
                                        max_score=10)

    x = np.linspace(-5, 15, 100)
    plt.plot(x, [sigmoid_map(i) for i in x])

    """

    linear_part = make_linear_part(max_lin_score, min_lin_score)
    bottom_part, min_ = make_bottom_part(bottom_base,
                                         max_lin_score, min_lin_score)
    top_part, max_ = make_top_part(top_base,
                                   max_lin_score, min_lin_score)
    if reverse:

        def ui_score_mapping(x):
            if x < min_lin_score:
                return max_score - max_score * (bottom_part(x) - min_) / (max_ - min_)
            if x > max_lin_score:
                return max_score - max_score * (top_part(x) - min_) / (max_ - min_)
            else:
                return max_score - max_score * (linear_part(x) - min_) / (max_ - min_)
    else:
        def ui_score_mapping(x):
            if x < min_lin_score:
                return max_score * (bottom_part(x) - min_) / (max_ - min_)
            if x > max_lin_score:
                return max_score * (top_part(x) - min_) / (max_ - min_)
            else:
                return max_score * (linear_part(x) - min_) / (max_ - min_)

    return ui_score_mapping


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


def between_percentiles_mean(scores, min_percentile=0.450, max_percentile=0.55):
    """
    Get the mean of the scores between the specified percentiles
    """
    import numpy
    scores = numpy.array(scores)
    sorted_scores = numpy.sort(scores)
    high_scores = sorted_scores[int(min_percentile * len(sorted_scores)): int(max_percentile * len(sorted_scores))]
    return numpy.mean(high_scores)


# numpy is imported here for compatibility with cloupickle serialization
import numpy


def tune_ui_map(scores,
                truth=None,
                all_normal=True,
                min_percentile_normal=0.25,
                max_percentile_normal=0.75,
                min_percentile_abnormal=0.25,
                max_percentile_abnormal=0.75,
                lower_base=10,
                upper_base=10):
    """
    Construct a ui scores map spreading out the scores between 0 and 10, where high means normal. Scores is
    an array of raw stroll scores.

    LOWERING the default range for the normal scores from [0.25, 0.75] to say [0., 0.25] will DECREASE the average
    quality score of normal sounds.

    INCREASING the range for the abnormal scores from [0.25, 0.75] to say [0.5, 1.0] will DECREASE the average quality
    score of abnormal sounds.
    """

    scores = numpy.array(scores)
    # we have examples of normal and abnormal
    if truth is not None and len(set(truth)) == 2:
        truth = np.array(truth)
        median_normal = between_percentiles_mean(scores[truth == 0], min_percentile=min_percentile_normal,
                                                 max_percentile=max_percentile_normal)
        median_abnormal = between_percentiles_mean(scores[truth == 1], min_percentile=min_percentile_abnormal,
                                                   max_percentile=max_percentile_abnormal)

    # if not the scores are all normal
    elif all_normal:
        median_normal = between_percentiles_mean(scores, min_percentile=min_percentile_normal,
                                                 max_percentile=max_percentile_normal)
        median_abnormal = 10 * median_normal

    # probably never useful, in case all scores are from abnormal
    else:
        median_abnormal = between_percentiles_mean(scores, min_percentile=min_percentile_abnormal,
                                                   max_percentile=max_percentile_abnormal)
        median_normal = median_abnormal / 10

    return median_normal, median_abnormal, lower_base, upper_base
