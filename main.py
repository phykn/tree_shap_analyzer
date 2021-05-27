from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from module.data_preparation import select_feature, split_data
from module.model_selection import select_model
from module.shap import get_shap_value, get_feature_importance
from module.simulation import simulator
import module.SessionState as SessionState
import warnings
warnings.filterwarnings(action='ignore')

# Config
class CFG:
    max_num_data = 10000
    max_num_feature = 30
    n_splits = 4
    max_shap_data_num = 100
    num_simulation = 1000


# Information
st.text('Tree model Analysis Tool | v0.6')

# Create Session
ss = SessionState.get(
    datas=None,
    output=None,
    is_model_selection=False,
    shap_value=None,
    importance=None,
)

# Define Function
@st.cache()
def load_data(filename, n=10000):
    # Load Data
    df = pd.read_csv(filename)
   
    # Sampling
    if len(df) > n:
        df = df.sample(n=n)

    # Select Column
    columns = df.columns
    columns = [column for column in columns if is_numeric_dtype(df[column])]
    df = df[columns]
    return df

# Load Data
filename = st.text_input('File path:', 'data/sample_data.csv')
df = load_data(filename, n=CFG.max_num_data)
columns = list(df.columns)

# Select Feature
st.text(f'Select Features (Maximum: {CFG.max_num_feature})')
feature_index = [st.checkbox(f'{column}', value=True) for column in columns]
features = list(np.array(columns)[feature_index])
st.text(f'Feature Number: {np.sum(feature_index)}')

# Select Models
model_list = st.multiselect(
    'Model',
    ['lgb', 'xgb', 'rf', 'et', 'gbr'],
    default=['lgb', 'xgb', 'rf', 'et', 'gbr'],
)

# Select Metric
metric = st.selectbox(
    'Metric',
    ['r2', 'mae', 'mse', 'rmse'],
    index=1
)

# Select Target
target = [st.selectbox('Target', [column for column in columns if column not in features])]

if st.button(f'Calculate ({target[0]})'):
    if np.sum(feature_index) > CFG.max_num_feature:
        st.text('Maximum number of features exceeded.')

    else:
        df_data = select_feature(df, features, target)
        datas = split_data(df_data, n_splits=CFG.n_splits)
        st.text(f'[{datetime.now()}] Done: Data separation.')

        output = select_model(model_list, datas, features, target, metric=metric)
        st.text(f'[{datetime.now()}] Done: Model selection.')

        shap_source, shap_value = get_shap_value(output['weights'], datas, features, max_num=CFG.max_shap_data_num)
        st.text(f'[{datetime.now()}] Done: SHAP Value calculation.')

        feature_names, feature_importances = get_feature_importance(shap_value, features, sort=True)

        ss.datas = datas
        ss.output = output
        ss.is_model_selection = True
        ss.shap_source = shap_source
        ss.shap_value = shap_value
        ss.importance = (feature_names, feature_importances)

# Plot Graph
if ss.is_model_selection:
    # Load Data
    output = ss.output
    feature_names, feature_importances = ss.importance

    # Plot Training Result
    true = np.array(output['oob_true'])
    pred = np.array(output['oob_pred'])
    r2 = np.corrcoef(true, pred)[0][1]**2
    mae = np.mean(np.abs(true-pred))
    minimum = np.minimum(np.min(true), np.min(pred))
    maximum = np.maximum(np.max(true), np.max(pred))

    fig, ax = plt.subplots()
    ax.set_title(f'Model: {output["model"]}, R$^{2}$: {r2:.4f}, MAE: {mae:.4f}')
    ax.scatter(true, pred, color='#EA4A54')
    ax.plot([minimum, maximum], [minimum, maximum], ls='--', lw=2, color='k')
    ax.set_xlim(minimum, maximum)
    ax.set_ylim(minimum, maximum)
    ax.set_xlabel('Ground Truth', fontsize=13)
    ax.set_ylabel('Prediction', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    st.pyplot(fig)

    # Plot Feature Importance
    num_init = np.minimum(10, len(feature_names))
    num = st.number_input(
        'Feature Number',
        value=num_init,
        min_value=1,
        max_value=len(feature_names),
        step=1
    )

    fig, ax = plt.subplots()
    ax.set_title(f'Target = {target[0]}')
    ax.barh(feature_names[:num][::-1], feature_importances[:num][::-1], color='#EA4A54')
    ax.set_xlim(left=0)
    ax.set_ylim(-1, num)
    ax.set_xlabel('Feature Importance (%)', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    st.pyplot(fig)

    # Plot Simulation
    graph = st.selectbox(
        'Graph', 
        ['SHAP', 'Simulation'],
        index=1
    )
    feature = st.selectbox(
        'Feature', 
        feature_names,
        index=0
    )
    index = np.where(np.array(features)==feature)[0][0]

    if graph == 'SHAP':        
        x = ss.shap_source[features[index]].values
        y = ss.shap_value[:, index]

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

        fig, ax = plt.subplots()

        ax.scatter(x, y, s=20, color='#EA4A54')
        ax.set_title(f'Target = {target[0]}')
        ax.bar(hist_x, hist_y, color='k', width=width, bottom=y_min-y_range*0.1, alpha=0.5)
        ax.set_ylim(bottom=y_min-y_range*0.1)
        ax.set_xlabel(f'{feature}', fontsize=13)
        ax.set_ylabel(f'SHAP Values for\n{feature}', fontsize=13)
        ax.tick_params(axis='both', labelsize=12)
        st.pyplot(fig)

    elif graph == 'Simulation':
        data = df[features[index]].dropna().values
        x_min, x_max = float(np.min(data)), float(np.max(data))

        x_set_min, x_set_max = st.slider(
            'Select a range of values',
            min_value=x_min, 
            max_value=x_max,
            value=(x_min, x_max),
            step=(x_max-x_min)/100
        )

        x, y = simulator(df, ss.output, features, feature, x_set_min, x_set_max, n=CFG.num_simulation)

        fig, ax = plt.subplots()
        ax.set_title(f'Target = {target[0]}')
        ax.scatter(x, y, color='#EA4A54')
        ax.set_xlim(x_set_min, x_set_max)
        ax.set_xlabel(f'{feature}', fontsize=13)
        ax.set_ylabel(f'Pred. {target[0]}', fontsize=13)
        ax.tick_params(axis='both', labelsize=12)
        st.pyplot(fig)