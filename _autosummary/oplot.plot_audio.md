# oplot.plot_audio

Functions intended to plot waveform and spectra plus some timestamps information as vertical lines on the plots

### Functions

| [`plot_lines`](#oplot.plot_audio.plot_lines)(ax, lines_loc[, label, color, ...])    | Function to draw vertical or horizontal lines on an ax   |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| [`plot_spectro`](#oplot.plot_audio.plot_spectro)(ax, wf[, chk_size, noverlap, sr])    |                                                          |
| [`plot_wf`](#oplot.plot_audio.plot_wf)(ax, wf[, wf_line_width, wf_color])        |                                                          |
| [`plot_wf_and_spectro`](#oplot.plot_audio.plot_wf_and_spectro)(wf[, figsize, chk_size, ...]) |                                                          |
| [`plot_wf_with_lines`](#oplot.plot_audio.plot_wf_with_lines)(wf[, figsize, sr, ...])        |                                                          |

### oplot.plot_audio.plot_lines(ax, lines_loc, label=None, color='r', line_width=0.5, line_style='-', line_type='vert', alpha=1)

Function to draw vertical or horizontal lines on an ax

* **Parameters:**
  * **ax** – the matplolib axis on which to draw
  * **lines_loc** – the location of the lines
  * **labels** – a list of floats, the labels of the lines, optionsl
  * **colors** – a list of strings, the colors of the lines
  * **line_widths** – a list of floats, the widths of the lines
  * **def_col** – default color if no list of colors if provided
  * **line_type** – ‘vert’ or ‘horiz

### Examples

An initial plot

fig, ax = plt.subplots()
… ax.plot([1, 2, 3])

Adding vertical lines to the plot

plot_vlines(ax,
… lines_loc=[1,2],
… line_type=’horiz’,
… colors=[‘b’, ‘r’],
… line_widths=[0.5, 2],
… labels=[‘thin blue’, ‘wide red’])

### oplot.plot_audio.plot_spectro(ax, wf, chk_size=2048, noverlap=0, sr=44100)

* **Parameters:**
  * **ax**
  * **wf**
  * **chk_size**
  * **noverlap**
  * **sr**

### oplot.plot_audio.plot_wf(ax, wf, wf_line_width=0.8, wf_color='b')

* **Parameters:**
  * **ax**
  * **wf**
  * **wf_line_width**
  * **wf_color**

### oplot.plot_audio.plot_wf_and_spectro(wf, figsize=(40, 8), chk_size=2048, noverlap=0, sr=44100, spectra_ylim=None, wf_y_lim=None, wf_x_lim=None, spectra_xlim=None, n_sec_per_tick=None, vert_lines_samp=None, vert_lines_sec=None, vert_lines_colors=None, vert_lines_labels=None, vert_lines_width=None, vert_lines_style=None, alpha_lines=None, n_tick_dec=None, wf_line_width=1, wf_color='b', title=None, title_font_size=10)

* **Parameters:**
  * **wf**
  * **figsize**
  * **chk_size**
  * **noverlap**
  * **sr**
  * **spectra_ylim**
  * **wf_y_lim**
  * **wf_x_lim**
  * **spectra_xlim**
  * **n_sec_per_tick**
  * **vert_lines_samp**
  * **vert_lines_sec**
  * **vert_lines_colors**
  * **vert_lines_labels**
  * **vert_lines_width**
  * **vert_lines_style**
  * **alpha_lines**
  * **n_tick_dec**
  * **wf_line_width**
  * **wf_color**
  * **title**
  * **title_font_size**
* **Returns:**

### oplot.plot_audio.plot_wf_with_lines(wf, figsize=(40, 10), sr=44100, wf_y_lim=None, wf_x_lim=None, n_sec_per_tick=None, vert_lines_samp=None, vert_lines_sec=None, vert_lines_colors=None, vert_lines_labels=None, vert_lines_width=None, vert_lines_style=None, alpha_lines=None, n_tick_dec=None, wf_line_width=1, wf_color='b', title=None, title_font_size=10)

* **Parameters:**
  * **wf**
  * **figsize**
  * **sr**
  * **wf_y_lim**
  * **wf_x_lim**
  * **n_sec_per_tick**
  * **vert_lines_samp**
  * **vert_lines_sec**
  * **vert_lines_colors**
  * **vert_lines_labels**
  * **vert_lines_width**
  * **vert_lines_style**
  * **alpha_lines**
  * **n_tick_dec**
  * **wf_line_width**
  * **wf_color**
  * **title**
  * **title_font_size**
* **Returns:**
