# oplot.ui_scores_mapping

Functions to create and plot outlier scores (or other) in a fixed bounded range. Intended to use to
show the results of an outlier algorithm in a user friendly UI

### Functions

| [`between_percentiles_mean`](#oplot.ui_scores_mapping.between_percentiles_mean)(scores[, ...])          | Get the mean of the scores between the specified percentiles                                                                                                                                                            |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`make_bottom_part`](#oplot.ui_scores_mapping.make_bottom_part)(base, max_score, min_score)     | The base has to be between 0 and 1, strictly.                                                                                                                                                                           |
| [`make_linear_part`](#oplot.ui_scores_mapping.make_linear_part)(max_score, min_score)           |                                                                                                                                                                                                                         |
| [`make_top_part`](#oplot.ui_scores_mapping.make_top_part)(base, max_score, min_score)        | The base has to be between 0 and 1, strictly.                                                                                                                                                                           |
| [`make_ui_score_mapping`](#oplot.ui_scores_mapping.make_ui_score_mapping)(min_lin_score, ...[, ...]) | Plot a sigmoid function to map outlier scores to (by default) the range (0, 10) The function is not only continuous but also smooth and the radius of the corners are controlled by the floats top_base and bottom_base |
| [`tune_ui_map`](#oplot.ui_scores_mapping.tune_ui_map)(scores[, truth, all_normal, ...])    | Construct a ui scores map spreading out the scores between 0 and 10, where high means normal.                                                                                                                           |

### oplot.ui_scores_mapping.between_percentiles_mean(scores, min_percentile=0.45, max_percentile=0.55)

Get the mean of the scores between the specified percentiles

### oplot.ui_scores_mapping.make_bottom_part(base, max_score, min_score)

The base has to be between 0 and 1, strictly.
The function will be of the form -base \*\* (-x + t) + C, where t and C
are the two constants to solve for. The constraints are continuity and
smoothness at max_score when pieced with the linear part

### oplot.ui_scores_mapping.make_linear_part(max_score, min_score)

* **Parameters:**
  * **bottom** – the proportion of the graph used for the bottom “sigmoid”
  * **middle** – the proportion of the graph used for the middle linear part
  * **top** – the proportion of the graph used for the top “sigmoid”
  * **max_score** – the maximum score seen on train
  * **min_score** – the minimum score seen on train
* **Returns:**
  the linear part of the ui score mapping

### oplot.ui_scores_mapping.make_top_part(base, max_score, min_score)

The base has to be between 0 and 1, strictly.
The function will be of the form -base \*\* (-x + t) + C, where t and C
are the two constants to solve for. The constraints are continuity and
smoothness at max_score when pieced with the linear part

### oplot.ui_scores_mapping.make_ui_score_mapping(min_lin_score, max_lin_score, top_base=2, bottom_base=2, max_score=10, reverse=False)

Plot a sigmoid function to map outlier scores to (by default) the range (0, 10)
The function is not only continuous but also smooth and the radius of the corners are controlled by the floats
top_base and bottom_base

* **Parameters:**
  * **min_lin_score** – float, the minimum scores which is map with a linear function
  * **max_lin_score** – float, the maximum scores which is map with a linear function
  * **top_base** – float, the base of the exponential function on top of the linear part
  * **bottom_base** – float, the base of the exponential function on the bottom of the linear part
  * **max_score** – float, the upper bound of the function
  * **reverse** – boolean, whether to mirror the function along its center
* **Returns:**
  a mapping, sigmoid like

———————— Example of use: —————————

from oplot.ui_scores_mapping import make_ui_score_mapping
import numpy as np
import matplotlib,pyplot as plt

sigmoid_map = make_ui_score_mapping(min_lin_score=1,
: max_lin_score=9,
  top_base=2,
  bottom_base=2,
  max_score=10)

x = np.linspace(-5, 15, 100)
plt.plot(x, [sigmoid_map(i) for i in x])

### oplot.ui_scores_mapping.tune_ui_map(scores, truth=None, all_normal=True, min_percentile_normal=0.25, max_percentile_normal=0.75, min_percentile_abnormal=0.25, max_percentile_abnormal=0.75, lower_base=10, upper_base=10, abnormal_fact=2)

Construct a ui scores map spreading out the scores between 0 and 10, where high means normal. Scores is
an array of raw stroll scores. NOTE: it assumes large scores means abnormal, small means normal!! Need to adapt
otherwise.

LOWERING the default range for the normal scores from [0.25, 0.75] to say [0., 0.25] will DECREASE the average
quality score of normal sounds.

INCREASING the range for the abnormal scores from [0.25, 0.75] to say [0.5, 1.0] will DECREASE the average quality
score of abnormal sounds.
