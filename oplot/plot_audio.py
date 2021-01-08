import matplotlib.pyplot as plt
# from cycler import cycler
# import numpy as np
# from matplotlib.pyplot import cm


def plot_lines(ax,
               lines_loc,
               label=None,
               color='r',
               line_width=0.5,
               line_style='-',
               line_type='vert'):
    """
    Function to draw vertical or horizontal lines on an ax
    :param ax: the matplolib axis on which to draw
    :param lines_loc: the location of the lines
    :param labels: a list of floats, the labels of the lines, optionsl
    :param colors: a list of strings, the colors of the lines
    :param line_widths: a list of floats, the widths of the lines
    :param def_col: default color if no list of colors if provided
    :param line_type: 'vert' or 'horiz
    :return:

    --------------EXAMPLE OF USAGE---------------

    # an initial plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    # adding vertical lines to the plot
    plot_vlines(ax,
                lines_loc=[1,2],
                line_type='horiz',
                colors=['b', 'r'],
                line_widths=[0.5, 2],
                labels=['thin blue', 'wide red'])
    """

    if line_type == 'vert':
        line_ = ax.axvline
    if line_type == 'horiz':
        line_ = ax.axhline
    for line in lines_loc:
        line_(line, c=color, linewidth=line_width, label=label, linestyle=line_style)
        # only the first one produce a label
        label = None


def plot_spectro(ax,
                 wf,
                 chk_size=2048,
                 noverlap=0,
                 sr=44100):
    ax.specgram(x=wf, NFFT=chk_size, noverlap=noverlap, Fs=sr)


def plot_wf(ax,
            wf,
            wf_line_width=0.8,
            wf_color='b'):
    ax.plot(wf, linewidth=wf_line_width, c=wf_color)


def plot_wf_and_spectro(wf,
                        figsize=(40, 8),
                        chk_size=2048,
                        noverlap=0,
                        sr=44100,
                        spectra_ylim=None,
                        wf_y_lim=None,
                        wf_x_lim=None,
                        spectra_xlim=None,
                        n_sec_per_tick=None,
                        vert_lines_samp=None,
                        vert_lines_sec=None,
                        vert_lines_colors=None,
                        vert_lines_labels=None,
                        vert_lines_width=None,
                        vert_lines_style=None,
                        n_tick_dec=None,
                        wf_line_width=1,
                        wf_color='b'):
    fig, ax = plt.subplots(2, 1, figsize=figsize)

    if n_tick_dec is None:
        n_tick_dec = max(str(n_sec_per_tick)[::-1].find('.'), 1)

    if n_sec_per_tick is None:
        # make a tick every 10% of the whole wf, roughly if possible, or every 1sec if 10% is less than 1sec
        n_sec_per_tick = max(int((len(wf) / sr) / 10), 1)

    # getting the ticks where we want them
    ticks_pos = range(0, len(wf), int(sr * n_sec_per_tick))
    ticks_labels = [f'{round(n_sec_per_tick * i, n_tick_dec)}s' for i in range(len(ticks_pos))]

    # TODO: udnerstand wtf is going on here
    plt.sca(ax[0])
    plt.xticks(ticks_pos, ticks_labels)
    plt.sca(ax[0])
    plt.xticks(ticks_pos, ticks_labels)

    # plot the wf
    plot_wf(ax[0], wf=wf, wf_line_width=wf_line_width, wf_color=wf_color)
    ax[0].set_xlim((0, len(wf)))
    # set some wf plot limits
    if wf_y_lim:
        ax[0].set_ylim(*wf_y_lim)
    if wf_x_lim:
        ax[0].set_xlim(*wf_x_lim)

    # plot the vertical lines:
    if vert_lines_samp is None:
        vert_lines_samp = []
    if vert_lines_sec is None:
        vert_lines_sec = []
    vert_lines_samp = list(vert_lines_samp)
    vert_lines_samp += [[int(i * sr) for i in list_lines] for list_lines in vert_lines_sec]

    for lines_idx, lines_loc in enumerate(vert_lines_samp):

        if vert_lines_labels is None:
            vert_line_label = None
        else:
            vert_line_label = vert_lines_labels[lines_idx]
        if vert_lines_colors is None:
            vert_lines_color = 'r'
        else:
            vert_lines_color = vert_lines_colors[lines_idx]
        if vert_lines_width is None:
            vert_line_width = 0.5
        else:
            vert_line_width = vert_lines_width[lines_idx]
        if vert_lines_style is None:
            vert_line_style = '-'
        else:
            vert_line_style = vert_lines_style[lines_idx]

        plot_lines(ax[0],
                   lines_loc=lines_loc,
                   label=vert_line_label,
                   color=vert_lines_color,
                   line_width=vert_line_width,
                   line_style=vert_line_style,
                   line_type='vert')
        first = False

    # plotting the spectrogram and some limits
    plot_spectro(ax=ax[1], wf=wf, chk_size=chk_size, noverlap=noverlap, sr=sr)
    if spectra_ylim:
        ax[1].set_ylim(*spectra_ylim)
    if spectra_xlim:
        ax[1].set_xlim(*spectra_xlim)

    plt.legend(loc=(1.04, 0.8))
    plt.show()
