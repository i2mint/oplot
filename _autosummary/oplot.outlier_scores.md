# oplot.outlier_scores

Visualizing various regions in a list of scores. E.g. adding colored regions
corresponding to percentiles to a timeline of outlier scores

### Functions

| [`find_last_normal_idx`](#oplot.outlier_scores.find_last_normal_idx)(sorted_truth)               | Return the index of the last 0 in the sorted_truth array                                                                                                                  |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`find_prop_markers`](#oplot.outlier_scores.find_prop_markers)(sorted_scores, sorted_truth)   | Find the score thresholds starting at which the proportion of n_normal / n_total is on or below the values in ratio_markers.                                              |
| [`get_confused_part`](#oplot.outlier_scores.get_confused_part)(sorted_scores, sorted_truth)   | Return the scores in the confused zone, i.e. scores whose range of values contains normal and anormal samples.                                                            |
| [`get_confusion_zones_percentiles`](#oplot.outlier_scores.get_confusion_zones_percentiles)(scores, truth)   | Get the percentiles of the normal scores in the confused zone.                                                                                                            |
| [`get_confusion_zones_std`](#oplot.outlier_scores.get_confusion_zones_std)(scores[, truth, ...])    | Get a list of zones boundaries based on the standard deviation of the normal scores                                                                                       |
| [`get_percentiles`](#oplot.outlier_scores.get_percentiles)(scores, n_percentiles)           | A function computing the n_percentiles of scores.                                                                                                                         |
| [`plot_scores_and_zones`](#oplot.outlier_scores.plot_scores_and_zones)(scores, zones[, box, ...]) | Plot the scores on a timeline with color according to which zone they belong too, green under the first value in zones, red above the last and a shade from green to red. |
| [`sort_scores_truth`](#oplot.outlier_scores.sort_scores_truth)(scores, truth)                 | Sort the aligned scores and truth arrays from lowest to largest                                                                                                           |

### oplot.outlier_scores.find_last_normal_idx(sorted_truth)

Return the index of the last 0 in the sorted_truth array

* **Parameters:**
  **sorted_truth** – an array of 0 or 1
* **Returns:**
  an int, the last idx of a 0 in sorted_truth

### oplot.outlier_scores.find_prop_markers(sorted_scores, sorted_truth, ratio_markers=(1, 0.75, 0.5), add_full_out_zone=True)

Find the score thresholds starting at which the proportion of n_normal / n_total
is on or below the values in ratio_markers. If the proportion is never reached, the
proportions are skipped and the thresholds array is shorter than the ratio_markers tuple
If add_full_out_zone is set to True, the zone where all scores are from abnormal sound is added and
any zone above that one is removed.

### oplot.outlier_scores.get_confused_part(sorted_scores, sorted_truth)

Return the scores in the confused zone, i.e. scores whose range of values contains normal and anormal samples.

* **Parameters:**
  * **sorted_scores** – an array of outlier scores, higher is more abnormal
  * **sorted_truth** – an array of 0 for normal and 1 for abnormal
* **Returns:**
  an array of scores

### oplot.outlier_scores.get_confusion_zones_percentiles(scores, truth, n_percentiles=1)

Get the percentiles of the normal scores in the confused zone.

* **Parameters:**
  * **scores** – an array of outlier scores
  * **truth** – an array of 0 for normal and 1 for abnormal
  * **n_percentiles** – the number of percentiles required
* **Returns:**
  an array of scores marking the boundary of the percentile zones

### oplot.outlier_scores.get_confusion_zones_std(scores, truth=None, n_zones=6, std_per_zone=0.5)

Get a list of zones boundaries based on the standard deviation of the normal scores

* **Parameters:**
  * **scores** – an array of outlier scores
  * **truth** – an array of 0 for normal and 1 for abnormal
  * **n_zones** – the number of zones required
  * **std_per_zone** – the number of standard deviation per zone
* **Returns:**
  an array of scores marking the boundary of the percentile zones

### oplot.outlier_scores.get_percentiles(scores, n_percentiles)

A function computing the n_percentiles of scores. If n_percentiles is larger than
len(scores), the scores are interpolated.

```pycon
>>> arr = [1, 2, 3, 4]
>>> get_percentiles(arr, n_percentiles=1)
array([3])
>>> get_percentiles(arr, n_percentiles=2)
array([2, 3])
>>> get_percentiles(arr, n_percentiles=3)
array([2, 3, 4])
>>> get_percentiles(arr, n_percentiles=4)
array([1, 2, 3, 4])
>>> get_percentiles(arr, n_percentiles=5)
array([1.5, 2. , 2.5, 3. , 3.5])
>>> get_percentiles(arr, n_percentiles=6)
array([1.5, 2. , 2.5, 3. , 3.5, 4. ])
>>> get_percentiles(arr, n_percentiles=7)
array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. ])
```

### oplot.outlier_scores.plot_scores_and_zones(scores, zones, box=None, title=None, lines=True)

Plot the scores on a timeline with color according to which zone they belong too, green under the first
value in zones, red above the last and a shade from green to red.

* **Parameters:**
  * **scores** – an array of scores
  * **zones** – the limit of the zones
  * **box** – limits to display the plot
  * **title** – name of the plot
  * **lines** – whether to show lines at the limit of the zones
* **Returns:**

### oplot.outlier_scores.sort_scores_truth(scores, truth)

Sort the aligned scores and truth arrays from lowest to largest
