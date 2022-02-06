import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from numpy import ndarray
from altair import Chart
from sklearn.metrics import confusion_matrix


def plot_reg_evaluation(
    true: ndarray, 
    pred: ndarray, 
    target: str=''
) -> Chart:
    minimum = np.minimum(np.min(true), np.min(pred))
    maximum = np.maximum(np.max(true), np.max(pred))

    source = pd.DataFrame(
        data = {'x': true,'y': pred}
    )

    base = alt.Chart(
        data = source, 
        title = f'Target = {target}'
    )

    fig_1 = base.mark_circle(
        size = 60, 
        color = '#EA4A54'
    ).encode(
        x = alt.X(
            'x:Q', 
            scale = alt.Scale(
                domain = (minimum, maximum)
            ), 
            title = 'Ground Truth'
        ),
        y = alt.Y(
            'y:Q', 
            scale = alt.Scale(
                domain = (minimum, maximum)
            ), 
            title = 'Prediction'
        ),
        tooltip = [
            alt.Tooltip(
                'x', 
                title = 'Ground Truth'
            ),
            alt.Tooltip(
                'y', 
                title = 'Prediction'
            )
        ]
    )

    fig_2 = base.mark_line(
        color = 'black', 
        strokeDash = [5,2]
    ).encode(
        x = alt.X(
            'x:Q', 
            scale = alt.Scale(
                domain=(minimum, maximum)
            )
        ),
        y = alt.Y(
            'x:Q', 
            scale = alt.Scale(
                domain=(minimum, maximum)
            )
        ),
    )

    fig = fig_1 + fig_2
    fig = fig.properties(
        width = 480,
        height = 480
    ).configure_title(
        fontSize = 20,
    ).configure_axis(
        labelFontSize = 15,
        titleFontSize = 20
    ).interactive()
    return fig


def plot_confusion_matrix(
    true: ndarray, 
    pred: ndarray, 
    target: str=''
) -> plt.figure:
    true = np.where(true > 0.5, 1, 0)
    pred = np.where(pred > 0.5, 1, 0)

    df_cm = pd.DataFrame(
        data = confusion_matrix(true, pred), 
        columns = np.unique(true), 
        index = np.unique(true)
    )
    df_cm.index.name = 'Ground Truth'
    df_cm.columns.name = 'Prediction'

    fig = plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4)
    plt.title(f'Target = {target}')    
    sns.heatmap(
        df_cm, 
        cmap = 'Blues', 
        annot = True, 
        annot_kws = {'size': 16}, 
        fmt = ''
    )
    return fig
