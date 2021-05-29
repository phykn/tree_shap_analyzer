import numpy as np

def add_histogram(ax, x, y):
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min

    y_min = np.min(y)
    y_max = np.max(y)
    y_range = y_max - y_min

    bins = np.arange(x_min, x_max, x_range*0.01)
    hist_y, hist_x = np.histogram(x, bins=bins)
    hist_x = (hist_x[:-1] + hist_x[1:]) / 2
    hist_y = np.minimum(hist_y, np.percentile(hist_y, 98))
    hist_y = 0.1 * y_range * hist_y / np.max(hist_y)
    width = np.mean(np.abs(hist_x[1:]-hist_x[:-1]))
    bottom = y_min-y_range*0.1
    ax.bar(hist_x, hist_y, color='k', width=width, bottom=bottom, alpha=0.3)
    return ax, bottom