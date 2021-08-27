import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.metrics import confusion_matrix
from .helper import Smoothing

def plot_training_result(true, pred, target_name):
    minimum = np.minimum(np.min(true), np.min(pred))
    maximum = np.maximum(np.max(true), np.max(pred))
    source  = pd.DataFrame({'x': true,'y': pred})
    base    = alt.Chart(source, title = f'Target = {target_name}')

    fig_1 = base.mark_circle(size=60, color='#EA4A54').encode(
        x = alt.X('x:Q', scale=alt.Scale(domain=(minimum, maximum)), title='Ground Truth'),
        y = alt.Y('y:Q', scale=alt.Scale(domain=(minimum, maximum)), title='Prediction'),
        tooltip = [alt.Tooltip('x', title='Ground Truth'),
                   alt.Tooltip('y', title='Prediction')]
    )

    fig_2 = base.mark_line(color='black', strokeDash=[5,2]).encode(
        x = alt.X('x:Q', scale=alt.Scale(domain=(minimum, maximum))),
        y = alt.Y('x:Q', scale=alt.Scale(domain=(minimum, maximum))),
    )

    fig = fig_1 + fig_2
    fig = fig.properties(
        width  = 480,
        height = 480
    ).configure_title(
        fontSize = 20,
    ).configure_axis(
        labelFontSize = 15,
        titleFontSize = 20
    ).interactive()
    return fig

def plot_confusion_matrix(true, pred, target_name):
    true = np.where(true > 0.5, 1, 0)
    pred = np.where(pred > 0.5, 1, 0)
    data = confusion_matrix(true, pred)
    df_cm = pd.DataFrame(data, columns=np.unique(true), index=np.unique(true))
    df_cm.index.name   = 'Ground Truth'
    df_cm.columns.name = 'Prediction'

    fig = plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)
    plt.title(f'Target = {target_name}')    
    sns.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={'size': 16}, fmt='')
    return fig

def plot_feature_importance(source, target_name, num=10, normalize=False):
    source = source.iloc[:num]
    base   = alt.Chart(source, title = f'Target = {target_name}')

    fig = base.mark_bar(color='#EA4A54').encode(
        x = alt.X('importance:Q', title = 'Importance' if normalize else 'Importance (%)'),
        y = alt.Y('feature:N',    title = None, sort='-x'),
        tooltip = [alt.Tooltip('feature', title='Feature'),
                   alt.Tooltip('importance', title='Importance')]
    )

    fig = fig.properties(
        width  = 480,
        height = 480
    ).configure_title(
        fontSize = 20,
    ).configure_axis(
        labelFontSize = 15,
        titleFontSize = 20
    )
    return fig

def plot_shap(x, y, x_hist, feature_name, target_name, mean):
    # Plot: Histogram
    fig_1, bottom = plot_histogram(x_hist, y)

    # Plot: Smoothing
    window_length = len(x)//5
    window_length = window_length + 1 if (window_length%2)==0 else window_length
    x_smooth, y_smooth = Smoothing(x, y, window_length=window_length)

    source = pd.DataFrame({'x': x_smooth,'y': y_smooth})
    base   = alt.Chart(source)

    fig_2 = base.mark_line(color='black').encode(
        x = 'x:Q',
        y = 'y:Q'
    )

    # Plot: SHAP
    source = pd.DataFrame({'x': x,'y': y})
    base   = alt.Chart(source, title = f'Target = {target_name}, MEAN = {mean}')

    fig_3 = base.mark_circle(size=60, color='#EA4A54').encode(
        x = alt.X('x:Q', scale=alt.Scale(domain=(np.min(x), np.max(x))), title=feature_name),
        y = alt.Y('y:Q', scale=alt.Scale(domain=(bottom, np.max(y))),    title=f'SHAP Value'),
        tooltip = [alt.Tooltip('x', title=feature_name),
                   alt.Tooltip('y', title='SHAP Value')]
    )

    fig = fig_3 + fig_2 + fig_1
    fig = fig.properties(
        width  = 480,
        height = 480
    ).configure_title(
        fontSize = 20,
    ).configure_axis(
        labelFontSize = 15,
        titleFontSize = 20
    ).interactive(bind_y=False)
    return fig

def plot_1d_simulation(x, y, x_hist, feature_name, target_name):
    # Plot: Histogram
    fig_1, bottom = plot_histogram(x_hist, y)

    # Plot: Line
    source = pd.DataFrame({'x': x,'y': y})
    base   = alt.Chart(source, title = f'Target = {target_name}')
    fig_2 = base.mark_line(color='#EA4A54').encode(
        x = alt.X('x:Q', scale=alt.Scale(domain=(np.min(x), np.max(x))), title=feature_name),
        y = alt.Y('y:Q', scale=alt.Scale(domain=(bottom, np.max(y))), title=f'Pred. {target_name}'),
        tooltip = [alt.Tooltip('x', title=feature_name),
                   alt.Tooltip('y', title=target_name)]
    )

    fig = fig_2 + fig_1
    fig = fig.properties(
        width  = 480,
        height = 480
    ).configure_title(
        fontSize = 20,
    ).configure_axis(
        labelFontSize = 15,
        titleFontSize = 20
    ).interactive(bind_y=False)
    return fig

def plot_2d_simulation(x1, x2, y, feature_name_1, feature_name_2, target_name):
    source = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    base   = alt.Chart(source, title = f'Target = {target_name}')
    fig = alt.Chart(source).mark_rect().encode(
        x = alt.X('x1:Q', bin=alt.Bin(maxbins=50), title=feature_name_1),
        y = alt.X('x2:Q', bin=alt.Bin(maxbins=50), title=feature_name_2),
        color = alt.Y('y:Q', scale=alt.Scale(scheme='redyellowblue', reverse=True), title=target_name),
        tooltip=[alt.Tooltip('x1', title=feature_name_1), 
                 alt.Tooltip('x2', title=feature_name_2), 
                 alt.Tooltip('y',  title=target_name)]
    )

    fig = fig.properties(
        width  = 480,
        height = 480
    ).configure_title(
        fontSize = 20,
    ).configure_axis(
        labelFontSize = 15,
        titleFontSize = 20
    )
    return fig

def plot_histogram(x, y):
    x_min   = np.min(x)
    x_max   = np.max(x)
    x_range = x_max - x_min
    bins    = np.arange(x_min, x_max, x_range*0.01)

    y_min = np.min(y)
    y_max = np.max(y)
    y_range = y_max - y_min
    
    hist_y, hist_x = np.histogram(x, bins=bins)
    hist_x = (hist_x[:-1] + hist_x[1:]) / 2
    hist_y = np.minimum(hist_y, np.percentile(hist_y, 98))
    bottom = y_min - y_range*0.1
    hist_y = 0.1 * y_range * hist_y / np.max(hist_y) + bottom  

    source = pd.DataFrame({'x': hist_x, 'y': hist_y, 'y_min': np.linspace(bottom, bottom, num=len(hist_x))})
    base   = alt.Chart(source)
    fig    = base.mark_area(color='lightgray').encode(x='x:Q', y='y_min:Q', y2='y:Q')
    return fig, bottom
