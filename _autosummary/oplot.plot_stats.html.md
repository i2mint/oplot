# oplot.plot_stats

Functions to represent the accuracy of outlier or classification algorithms

### Functions

| [`apply_function_on_consecutive`](#oplot.plot_stats.apply_function_on_consecutive)(scores, ...[, ...])   | Apply func to each block of consecutive value in scores if the corresponding values in arr_for_consec are constant                                                               |
|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`area_under_points`](#oplot.plot_stats.area_under_points)(points)                           | Given a list of pair corresponding to the x and y coordinates of the points, find the area under the curve of the (simplest) piecewise linear function passing by all the points |
| [`cumulative_tn_fp_fn_tp`](#oplot.plot_stats.cumulative_tn_fp_fn_tp)(truth, scores)               | Compute efficiently the cumulative tn, fp, fn and tp                                                                                                                             |
| [`get_tn_fp_fn_tp`](#oplot.plot_stats.get_tn_fp_fn_tp)(truth, scores[, threshold])         | compute the counts of true negative, false positive, false negative and true positive as predicted by the outlier score for the given threshold                                  |
| [`list_mult`](#oplot.plot_stats.list_mult)(l, mult[, random_remainder])              | Extend the multiplication of lists by an integer naturally to float.                                                                                                             |
| [`make_heatmap`](#oplot.plot_stats.make_heatmap)(matrix_results, tags[, ...])           | Makes a heatmap plot of the matrix_results where the entries are rounded.                                                                                                        |
| [`make_normal_outlier_timeline`](#oplot.plot_stats.make_normal_outlier_timeline)(y, scores[, ...])      | Plots all scores grouped by their y values in order specified by y_order or np.unique(y) is left to None.                                                                        |
| [`make_tables_tn_fp_fn_tp`](#oplot.plot_stats.make_tables_tn_fp_fn_tp)(truth, scores[, ...])       | Make a table of counts of tn, fp, fn, tp for n_thresholds equally spaced in threshold_range                                                                                      |
| [`make_tn_fp_fn_tp_tag_lists`](#oplot.plot_stats.make_tn_fp_fn_tp_tag_lists)(truth, scores, ...)      | Return a dictionary with the keys being tp, tn, fp and fn and the corresponding values are the indices of the scores/truth/tags for the given threshold.                         |
| [`pair_metrics_to_reference`](#oplot.plot_stats.pair_metrics_to_reference)([pair_metrics, ...])      | Utility to compute the reference/chance curve for a pair metrics type curve.                                                                                                     |
| [`parallel_sort`](#oplot.plot_stats.parallel_sort)(iterable_list[, sort_idx])            | Sort several lists in iterable_list in parallel, according to the the list of index sort_idx                                                                                     |
| [`pick_equally_spaced_points`](#oplot.plot_stats.pick_equally_spaced_points)(values, n_points)        |                                                                                                                                                                                  |
| [`plot_confusion_matrix`](#oplot.plot_stats.plot_confusion_matrix)(y_true, y_pred[, fig, ...])   | This function prints and plots the confusion matrix.                                                                                                                             |
| [`plot_freqs_stats`](#oplot.plot_stats.plot_freqs_stats)(X[, upper_frequency, ...])         | X is intended to be the list/array of spectra, the function plots the mean, max and min of each frequency.                                                                       |
| [`plot_outlier_metric_curve`](#oplot.plot_stats.plot_outlier_metric_curve)(truth, scores[, ...])     | Plots one outlier scores metric against another one.                                                                                                                             |
| [`rebalance_scores`](#oplot.plot_stats.rebalance_scores)(test_scores, test_truth, ...)      | Re-balances the ratio of normal/outlier scores by copying the normal/outliers scores when needed.                                                                                |
| [`rebalancing_normal_outlier_ratio`](#oplot.plot_stats.rebalancing_normal_outlier_ratio)(...)               | Rebalance artificially the ratio outlier/(normal + outlier) to the specified percent_outliers.                                                                                   |
| [`render_mpl_table`](#oplot.plot_stats.render_mpl_table)(data[, col_width, ...])            | Take a pandas dataframe and represents it with a picture.                                                                                                                        |
| [`smooth_scores`](#oplot.plot_stats.smooth_scores)(scores[, window_size, ...])           | Smooth an iterable of score by applying smooth_funct to each window of size window_size.                                                                                         |
| [`split_on_consecutive`](#oplot.plot_stats.split_on_consecutive)(arr_to_split, ...)             | Split arr_to_split based on the entries of arr_for_consec.                                                                                                                       |
| [`vlines`](#oplot.plot_stats.vlines)(x[, ymin, ymax, marker, ...])                | Plot vlines in a more intuitive way than the default matplotlib version                                                                                                          |
| [`wiggle_scores`](#oplot.plot_stats.wiggle_scores)(scores, truth)                        | Sort scores from low to high while keeping truth aligned with it.                                                                                                                |
| [`wiggle_values_keep_order`](#oplot.plot_stats.wiggle_values_keep_order)(values)                    | Wiggles the values in a list of scores so has to remove any duplicate values while keeping the same order.                                                                       |

### oplot.plot_stats.apply_function_on_consecutive(scores, arr_for_consec, func=<function mean>)

> Apply func to each block of consecutive value in scores if the corresponding values in arr_for_consec are constant
* **Parameters:**
  * **scores** – a list of floats
  * **arr_for_consec** – an array of values defining the blocks
* **Returns:**
  a list the output of func on each block

### oplot.plot_stats.area_under_points(points)

Given a list of pair corresponding to the x and y coordinates of the points, find the
area under the curve of the (simplest) piecewise linear function passing by all the points

```pycon
>>> area_under_points([(0, 1), (1, 1)])
1.0
>>> area_under_points([(0, 1), (1, 2)])
1.5
>>> area_under_points([(0, 1), (1, 2), (2, 2)])
3.5
```

### oplot.plot_stats.cumulative_tn_fp_fn_tp(truth, scores)

Compute efficiently the cumulative tn, fp, fn and tp

### oplot.plot_stats.get_tn_fp_fn_tp(truth, scores, threshold=2)

compute the counts of true negative, false positive, false negative and true positive as predicted
by the outlier score for the given threshold

* **Parameters:**
  * **truth** – a list of 0 and 1’s, 0 meaning normal and 1 meaning outlier
  * **scores** – a list of outlier scores, of same length as truth
  * **threshold** – a float, any score under is predicted as normal and above as outlier
* **Returns:**
  true negative, false positive, false negative, true positive

### oplot.plot_stats.list_mult(l, mult, random_remainder=False)

Extend the multiplication of lists by an integer naturally to float.
EX: [1,2] \* 1.5 = [1,2,1]
If random_remainder is set to True, the decimal part of the new list will be chose at random from l.

### oplot.plot_stats.make_heatmap(matrix_results, tags, rounding=4, fig_size=(20, 20), make_symmetric=False, fill_diag=None, cmap=<matplotlib.colors.LinearSegmentedColormap object>, name='')

Makes a heatmap plot of the matrix_results where the entries are rounded.
If matrix_results is upper or lower diagonal and make_symmetric is set to true,
plot will be made symmetric by copying the strict upper or lower half.

### oplot.plot_stats.make_normal_outlier_timeline(y, scores, y_order=None, vertical_sep=False, saving_path=None, fig_size=(16, 5), name='normal/outlier scores', smooth=False, legend_size=10, title_font_size=10, label_for_y=None, legend_n_cols=1, xticks=None, xticks_labels=None, xticks_rotation=90)

Plots all scores grouped by their y values in order specified by y_order or np.unique(y) is left to None.

* **Parameters:**
  * **scores** – an array of outlier scores
  * **y** – an array of tags, each tag will get its own color on the plot and its name on the legend
    The vertical line will also be grouped according to the tags, in the order given in y_order
  * **fig_size** – the size of the plot
  * **line_width** – the thickness of the line on the plot
* **Returns:**

### oplot.plot_stats.make_tables_tn_fp_fn_tp(truth, scores, threshold_range=None, n_thresholds=10, normalize=False)

Make a table of counts of tn, fp, fn, tp for n_thresholds equally spaced in threshold_range

* **Parameters:**
  * **truth** – a list of 0 and 1’s, 0 meaning normal and 1 meaning outlier
  * **scores** – a list of outlier scores, of same length as truth
  * **threshold_range** – two values in a tuple or list. The thresholds will be picked within these.
  * **n_thresholds** – the number of thresholds to produce
* **Returns:**
  a panda dataframe with rows of the form (threshold, tn, tp, fn, fp)

### oplot.plot_stats.make_tn_fp_fn_tp_tag_lists(truth, scores, threshold, tags=None)

Return a dictionary with the keys being tp, tn, fp and fn and the corresponding values are the indices of the scores/truth/tags
for the given threshold. If a list of tags of the sounds corresponding to each scores is given, the values of the dict will
be those tags rather than indices

### oplot.plot_stats.pair_metrics_to_reference(pair_metrics={'x': 'TPR', 'y': 'FPR'}, outlier_proportion=0.2, label='chance line', base_statistics_dict={'ACC': <function <lambda>>, 'BM': <function <lambda>>, 'F1': <function <lambda>>, 'FDR': <function <lambda>>, 'FNR': <function <lambda>>, 'FOR': <function <lambda>>, 'FPR': <function <lambda>>, 'MK': <function <lambda>>, 'NMCC': <function <lambda>>, 'NPV': <function <lambda>>, 'PPV': <function <lambda>>, 'TNR': <function <lambda>>, 'TPR': <function <lambda>>, 'TS': <function <lambda>>}, synonyms={'ACC': ['accuracy', 'acc'], 'BM': ['informedness', 'bookmaker_informedness', 'bi', 'BI', 'bm'], 'F1': ['f1_score', 'f1', 'F1_score'], 'FDR': ['false_discovery_rate', 'fdr'], 'FNR': ['miss_rate', 'false_negative_rate', 'fnr'], 'FOR': ['false_omission_rate', 'for'], 'FPR': ['fall_out', 'false_positive_rate', 'fpr'], 'MK': ['markedness', 'mk'], 'NMCC': ['normalized_Matthews_correlation_coefficient', 'nmcc'], 'NPV': ['negative_predictive_value', 'npv'], 'PPV': ['precision', 'positive_predictive_value', 'ppv'], 'TNR': ['specificity', 'SPC', 'true_negative_rate', 'selectivity', 'tnr'], 'TPR': ['recall', 'sensitivity', 'true_positive_rate', 'hit_rate', 'tpr'], 'TS': ['threat_score', 'critical_success_index', 'CSI', 'csi', 'ts']})

Utility to compute the reference/chance curve for a pair metrics type curve. Note that for certain combination
of metrics, the curve may be a point.

### oplot.plot_stats.parallel_sort(iterable_list, sort_idx=0)

Sort several lists in iterable_list in parallel, according to the the list of index sort_idx

* **Parameters:**
  * **iterable_list** – list of list, all the lists have the same length
  * **sort_idx** – int, the index of the list to sort by
* **Returns:**
  a list sorted tuples

```pycon
>>> parallel_sort([[2, 3, 1], ['a', 'b', 'c']])
[(1, 2, 3), ('c', 'a', 'b')]
```

```pycon
>>> parallel_sort([[2, 3, 1], ['a', 'b', 'c'], [10, 9, 7]], sort_idx=2)
[(1, 3, 2), ('c', 'b', 'a'), (7, 9, 10)]
```

### oplot.plot_stats.pick_equally_spaced_points(values, n_points)

* **Parameters:**
  * **values** – a list of values
  * **n_points** – the number of indices to choose
* **Returns:**
  a list of indices to choose from the list values in order for these picked values
  to be as equally spaced as possible, starting with the first value and ending with the final value

DO NOT TRUST THIS FUNCTION! It is made only for increasing values and is only somewhat accurate if the values
are “fine” compared to the number of point n_points. It works well for what it is intended: finding equally spaced
points on a precision/recall curve if many points are available on the curve.

```pycon
>>> pick_equally_spaced_points([1,2,3], 3)
array([0, 1, 2])
>>> pick_equally_spaced_points(range(300), 3)
array([  0, 150, 299])
>>> pick_equally_spaced_points([1, 1.1, 1.2, 4, 5, 6], 3)
array([0, 3, 5])
```

### oplot.plot_stats.plot_confusion_matrix(y_true, y_pred, fig=None, ax=None, classes=None, normalize=False, title=None, cmap=<matplotlib.colors.LinearSegmentedColormap object>, saving_path=None, figsize=(10, 10), color_bar=False, plot=True, cm=False)

This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.

### oplot.plot_stats.plot_freqs_stats(X, upper_frequency=22050, n_bins=1025, normalized=True)

X is intended to be the list/array of spectra, the function plots the mean, max and min of each frequency.
If normalized, the min/max/mean entries are all divided by the the minimum value in the mean array.

### oplot.plot_stats.plot_outlier_metric_curve(truth, scores, pair_metrics={'x': 'TPR', 'y': 'PPV'}, plot_curve=True, curve_legend_name=None, title=None, plot_table_points_on_curve=False, plot_chance_line=True, plot_table=False, n_points_for_table=10, axis_name_dict=None, saving_root=None, outlier_proportion=None, wiggle=False, table_dpi=300, base_statistics_dict={'ACC': <function <lambda>>, 'BM': <function <lambda>>, 'F1': <function <lambda>>, 'FDR': <function <lambda>>, 'FNR': <function <lambda>>, 'FOR': <function <lambda>>, 'FPR': <function <lambda>>, 'MK': <function <lambda>>, 'NMCC': <function <lambda>>, 'NPV': <function <lambda>>, 'PPV': <function <lambda>>, 'TNR': <function <lambda>>, 'TPR': <function <lambda>>, 'TS': <function <lambda>>}, synonyms={'ACC': ['accuracy', 'acc'], 'BM': ['informedness', 'bookmaker_informedness', 'bi', 'BI', 'bm'], 'F1': ['f1_score', 'f1', 'F1_score'], 'FDR': ['false_discovery_rate', 'fdr'], 'FNR': ['miss_rate', 'false_negative_rate', 'fnr'], 'FOR': ['false_omission_rate', 'for'], 'FPR': ['fall_out', 'false_positive_rate', 'fpr'], 'MK': ['markedness', 'mk'], 'NMCC': ['normalized_Matthews_correlation_coefficient', 'nmcc'], 'NPV': ['negative_predictive_value', 'npv'], 'PPV': ['precision', 'positive_predictive_value', 'ppv'], 'TNR': ['specificity', 'SPC', 'true_negative_rate', 'selectivity', 'tnr'], 'TPR': ['recall', 'sensitivity', 'true_positive_rate', 'hit_rate', 'tpr'], 'TS': ['threat_score', 'critical_success_index', 'CSI', 'csi', 'ts']}, return_rauc=True, add_point_left=None, add_point_right=None)

Plots one outlier scores metric against another one. The metrics name can be any names in the base_statistics_dict
or the synonyms dict. The chance line/curve is automatically computed and displayed along with a table
of equally spaced point on the curve.

* **Parameters:**
  * **truth** – an array of 0/1, the ground truth: 0 for normal, 1 for outlier
  * **scores** – the scores as predicted by our model. Higher scores is expected to correspond to outliers.
  * **pair_metrics** – A dictionary with two keys, one for each of the metrics to represent. (x/y on the x/y axis)
  * **plot_curve** – boolean, whether to plot the curve
  * **curve_legend_name** – the name of the curve as displayed in the legend
  * **title** – the title of the curve, by default the name of the metrics
  * **plot_table_points_on_curve** – whether to display dots on the rauc curve corresponding to points in the table
  * **plot_chance_line** – boolean, whether or not the display the chance line
  * **plot_table** – boolean, whether or not to plot the table, useful to share nice pics with customers
  * **n_points_for_table** – int, the number of equally spaced points for the table
  * **axis_name_dict** – a dictionary specifying the name to display on the x/y axis. If set to None, the names in
    pair_metrics are used
  * **saving_root** – if set, path to the folder where the pictures will be saved.
  * **outlier_proportion** – None or a float between 0 and 1. If a float is chosen either the normal scores or the
    anomaly scores will be copied over to achieve the requested proportion of outlier
  * **wiggle** – boolean, whether the scores will be slightly modified in order to avoid duplicate. Can
    help in drawing the curve when too many scores are the same. Most often
    the problem wiggle_scores solves arise from using sklearn OneClassSVM for the scores
    computation
  * **table_dpi** – int, the higher the finer the pic
  * **base_statistics_dict** – the dictionary of possible metrics with the functions computing the metrics from
    the tn_fp_fn_tp counts. See above.
  * **synonyms** – a dictionary containing the synonymous, allowing the user to refer to the metrics in different
    terms
  * **return_rauc** – boolean, whether or not to return the area under the curve

### oplot.plot_stats.rebalance_scores(test_scores, test_truth, outlier_proportion)

Re-balances the ratio of normal/outlier scores by copying the normal/outliers scores when needed.
This is useful to compute real life precision/recall and other such metrics when the actual proportion
of outlier/normal is known but not achieved in the test set.

### oplot.plot_stats.rebalancing_normal_outlier_ratio(normal_scores, outlier_scores, percent_outliers)

Rebalance artificially the ratio outlier/(normal + outlier) to the specified percent_outliers.
Does this by copying data points, use with caution!

* **Parameters:**
  * **normal_scores**
  * **outlier_scores**
  * **percent_outliers**
* **Returns:**

### oplot.plot_stats.render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14, header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w', bbox=[0, 0, 1, 1], header_columns=0, ax=None, path_to_save=None, round_decimals=3, cols_to_round=(), cols_to_int='all_other', dpi=300, \*\*kwargs)

Take a pandas dataframe and represents it with a picture. This allows to save a .png version of the dataframe.

### oplot.plot_stats.smooth_scores(scores, window_size=2, window_step=None, smooth_func=<function mean>)

Smooth an iterable of score by applying smooth_funct to each window of size window_size.
If scores is smaller than window_size, an empty list is returned.

* **Parameters:**
  * **scores** – list, the scores to smooth
  * **window_size** – int, the size of the window
  * **smooth_funct** – function, the function applied to the windows
* **Returns:**
  a new list of scores

```pycon
>>> list(smooth_scores([1], window_size=2))
[]
>>> list(smooth_scores([1, 2], window_size=2))
[np.float64(1.5)]
>>> list(smooth_scores([1, 2, 3], window_size=2, window_step=1, smooth_func=np.max))
[np.int64(2), np.int64(3)]
```

### oplot.plot_stats.split_on_consecutive(arr_to_split, arr_for_consec)

Split arr_to_split based on the entries of arr_for_consec. Each segment of consecutive
equal entries in arr_for_consec will induce a split

```pycon
>>> split_on_consecutive([1, 1, 2], [1, 1, 2])
[array([1, 1]), array([2])]
>>> split_on_consecutive(['a', 'b', 'c', 'd', 'e', 'f', 'g'], [1, 1, 2, 1, 1, 1, 3])
[array(['a', 'b'], dtype='<U1'), array(['c'], dtype='<U1'), array(['d', 'e', 'f'], dtype='<U1'), array(['g'], dtype='<U1')]
```

### oplot.plot_stats.vlines(x, ymin=0, ymax=None, marker='o', marker_kwargs=None, colors='k', linestyles='solid', label='', data=None, \*\*kwargs)

Plot vlines in a more intuitive way than the default matplotlib version

### oplot.plot_stats.wiggle_scores(scores, truth)

Sort scores from low to high while keeping truth aligned with it. The original values
in scores which are present multiple times are spread out equally between their value and
the next larger score.

* **Parameters:**
  * **scores** – a list of scores
  * **truth** – a list of 0/1, for normal/abnormal
* **Returns:**
  scores and truth list sorted from low to high score and where the scores have been
  wiggled

```pycon
>>> wiggle_scores([1, 1, 2], [0, 0, 0])
(array([1. , 1.5, 2. ]), array([0, 0, 0]))
```

```pycon
>>> wiggle_scores([4, 1, 1, 2], [1, 0, 0, 0])
(array([1. , 1.5, 2. , 4. ]), array([0, 0, 0, 1]))
```

### oplot.plot_stats.wiggle_values_keep_order(values)

Wiggles the values in a list of scores so has to remove any duplicate values while keeping the same order.
This is intended to use with plot_outlier_metric_curve in order to smooth the curve in situations where scores
are repeating a lot

```pycon
>>> scores = [1, 1, 2, 3, 3, 4, 5, 5]
>>> wiggle_values_keep_order(scores)
array([1. , 1.5, 2. , 3. , 3.5, 4. , 5. , 5.5])
```

* **Parameters:**
  **values** – a list of floats in general
* **Returns:**
  a new list of float, where the values have been moved around a little, without altering their order
