import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.axes._subplots import Subplot
from .explanation import smoothing


def template(
    ax: Subplot,
    title: Optional[str]=None,
    xlabel: Optional[str]=None,
    ylabel: Optional[str]=None,
    xlim: Tuple[Optional[float]]=(None, None),
    ylim: Tuple[Optional[float]]=(None, None),
) -> Subplot:
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis="both", labelsize=12),
    ax.set_facecolor((0.97, 0.97, 0.97))
    ax.grid(visible=True, which="major", axis="both", color="k", alpha=0.12)
    return ax


def plot_histogram(
    ax: Subplot, 
    x: ndarray, 
    y: ndarray
) -> Tuple[Union[Subplot, float]]:
    # x range
    x_min = np.min(x)
    x_max = np.max(x)
    if x_min == x_max:
        return None, None

    x_range = x_max - x_min
    bins = np.arange(x_min, x_max, x_range * 0.01)

    # y range
    y_min = np.min(y)
    y_max = np.max(y)
    y_range = y_max - y_min

    # bins
    hist_y, hist_x = np.histogram(x, bins=bins)
    hist_x = (hist_x[:-1] + hist_x[1:]) / 2
    hist_y = np.minimum(hist_y, np.percentile(hist_y, 98))
    bottom = y_min - y_range * 0.1
    hist_y = 0.1 * y_range * hist_y / np.max(hist_y) + bottom

    # ax
    width = np.mean(np.abs(hist_x[1:]-hist_x[:-1]))
    ax.plot(hist_x, hist_y, color="lightgray", alpha=0.5)
    ax.fill_between(
        hist_x, 
        hist_y, 
        np.linspace(bottom, bottom, num=len(hist_x)), 
        color="lightgray"
    )
    return ax, bottom


def plot_simulation_1d(
    x: ndarray, 
    y: ndarray, 
    x_all: ndarray, 
    feature: str, 
    target: str
) -> Figure:
    # initialize
    fig, ax = plt.subplots()

    # histogram
    hist_ax, bottom = plot_histogram(ax, x_all, y)
    if hist_ax is not None:
        ax = hist_ax

    # line
    ax.plot(x, y, color="#EA4A54")

    # apply template
    ax = template(
        ax,
        title=f"Target = {target}",
        xlabel=feature,
        ylabel="Prediction",
        xlim=(np.min(x), np.max(x)),
        ylim=(bottom, None)
    )
    return fig


def plot_shap(
    x: ndarray, 
    y: ndarray,
    x_all: ndarray,
    feature: str,
    target: str,
    mean: float
) -> Figure:
    # initialize
    fig, ax = plt.subplots()

    # histogram
    hist_ax, bottom = plot_histogram(ax, x_all, y)
    if hist_ax is not None:
        ax = hist_ax

    # shap    
    ax.scatter(x, y, color="#EA4A54")

    # smoothing
    window_length = len(x)//5
    window_length = window_length + 1 if (window_length%2)==0 else window_length
    x_smooth, y_smooth = smoothing(x, y, window_length=window_length)
    ax.plot(x_smooth, y_smooth, color="k")

    # apply template
    ax = template(
        ax,
        title=f"Target = {target}, MEAN = {mean:.2f}",
        xlabel=feature,
        ylabel="SHAP value",
        xlim=(np.min(x), np.max(x)),
        ylim=(bottom, None)
    )
    return fig