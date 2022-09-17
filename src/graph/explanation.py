import numpy as np
import pandas as pd
import altair as alt
from typing import Tuple
from numpy import ndarray
from altair import Chart
from scipy.signal import savgol_filter


def plot_shap(
    x: ndarray, 
    y: ndarray,
    x_all: ndarray,
    feature: str,
    target: str,
    mean: float
) -> Chart:
    # histogram
    fig_1, bottom = plot_histogram(x_all, y)

    # smoothing
    window_length = len(x)//5
    window_length = window_length+1 if (window_length%2)==0 else window_length
    x_smooth, y_smooth = smoothing(x, y, window_length=window_length)

    source = pd.DataFrame(dict(x=x_smooth, y=y_smooth))
    base = alt.Chart(source)
    fig_2 = base.mark_line(color="black").encode(x="x:Q", y="y:Q")

    # Plot: SHAP
    source = pd.DataFrame(dict(x=x, y=y))
    base = alt.Chart(data=source, title=f"Target = {target}, MEAN = {mean:.2f}")

    fig_3 = base.mark_circle(size=60, color="#EA4A54"
    ).encode(
        x = alt.X(
            "x:Q", 
            scale=alt.Scale(domain=(np.min(x), np.max(x))), 
            title=feature
        ),
        y = alt.Y(
            "y:Q", 
            scale=alt.Scale(domain=(bottom, np.max(y))),
            title="SHAP Value"
        ),
        tooltip = [
            alt.Tooltip("x", title=feature),
            alt.Tooltip("y", title="SHAP Value")
        ]
    )

    if fig_1 is None:
        fig = fig_3 + fig_2
    else:
        fig = fig_3 + fig_2 + fig_1
    fig = fig.properties(
        width=480,
        height=480
    ).configure_title(
        fontSize=20,
    ).configure_axis(
        labelFontSize=15,
        titleFontSize=20
    ).interactive(bind_x=True, bind_y=True)
    return fig


def plot_simulation_1d(
    x: ndarray, 
    y: ndarray, 
    x_all: ndarray, 
    feature: str, 
    target: str
) -> Chart:
    # histogram
    fig_1, bottom = plot_histogram(x_all, y)

    # line
    source = pd.DataFrame(dict(x=x, y=y))
    base = alt.Chart(data=source, title=f"Target = {target}")

    fig_2 = base.mark_line(color="#EA4A54"
    ).encode(
        x = alt.X(
            "x:Q", 
            scale=alt.Scale(domain=(np.min(x), np.max(x))), 
            title=feature
        ),
        y = alt.Y(
            "y:Q", 
            scale=alt.Scale(domain=(bottom, np.max(y))), 
            title="Prediction"
        ),
        tooltip = [
            alt.Tooltip("x", title=feature),
            alt.Tooltip("y", title=target)
        ]
    )

    fig = fig_2 + fig_1
    fig = fig.properties(
        width=480,
        height=480
    ).configure_title(
        fontSize=20,
    ).configure_axis(
        labelFontSize=15,
        titleFontSize=20
    ).interactive(bind_x=True, bind_y=True)
    return fig


def plot_simulation_2d(
    x_0: ndarray, 
    x_1: ndarray, 
    y: ndarray, 
    feature_0: str,
    feature_1: str,
    target: str
) -> Chart:
    source = pd.DataFrame(dict(x_0=x_0, x_1=x_1, y=y))    
    fig = alt.Chart(source).mark_rect().encode(
        x = alt.X("x_0:Q", bin=alt.Bin(maxbins=50), title=feature_0),
        y = alt.X("x_1:Q", bin=alt.Bin(maxbins=50), title=feature_1),
        color = alt.Y(
            "y:Q", 
            scale=alt.Scale(scheme="redyellowblue", reverse=True), 
            title=target
        ),
        tooltip=[
            alt.Tooltip("x_0", title=feature_0),
            alt.Tooltip("x_1", title=feature_1), 
            alt.Tooltip("y", title=target)
        ]
    )

    fig = fig.properties(
        width=480,
        height=480
    ).configure_title(
        fontSize=20,
    ).configure_axis(
        labelFontSize=15,
        titleFontSize=20
    )
    return fig


def plot_histogram(
    x: ndarray, 
    y: ndarray
) -> Tuple[Chart, float]:
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    bins = np.arange(x_min, x_max, x_range * 0.01)

    y_min = np.min(y)
    y_max = np.max(y)
    y_range = y_max - y_min
    
    hist_y, hist_x = np.histogram(x, bins=bins)
    hist_x = (hist_x[:-1] + hist_x[1:]) / 2
    hist_y = np.minimum(hist_y, np.percentile(hist_y, 98))
    bottom = y_min - y_range * 0.1
    hist_y = 0.1 * y_range * hist_y / np.max(hist_y) + bottom  

    source = pd.DataFrame(
        data = dict(
            x=hist_x, 
            y=hist_y, 
            y_min=np.linspace(bottom, bottom, num=len(hist_x))
        )
    )
    base = alt.Chart(source)

    fig = base.mark_area(color="lightgray").encode(x="x:Q", y="y_min:Q", y2="y:Q")
    return fig, bottom


def smoothing(
    x: ndarray, 
    y: ndarray, 
    window_length: int=9, 
    polyorder: int=2
) -> Tuple[ndarray, ndarray]:
    assert len(x)==len(y), "Error. Not same length."

    index = np.argsort(x)
    x = x[index]
    y = y[index]

    y_length = len(y)    
    if y_length <= polyorder:
        out = np.full_like(y, np.mean(y))
    else:
        if y_length < window_length:
            if y_length % 2 == 0:
                window_length = y_length - 1
            else:
                window_length = y_length
                
        if window_length <= polyorder:
            out = np.full_like(y, np.mean(y))
        else:
            out = savgol_filter(y, window_length=window_length, polyorder=polyorder)

    return x, out