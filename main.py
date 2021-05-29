# Import Library
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import gc
from glob import glob
from datetime import datetime
from pandas.api.types import is_numeric_dtype
from module.metrics import r2_score, auc_score, accuracy_score
from module.data_preparation import select_feature, split_data, del_outlier
from module.model_selection import select_model
from module.shap import get_feature_importance, get_shap_value_from_output, get_important_feature
from module.simulation import simulator
from module.helper import add_histogram
import module.SessionState as SessionState
import warnings
warnings.filterwarnings(action='ignore')

# Head
st.markdown('# Model Analyzer ver 0.7')

# Config
class CFG:
    n_jobs = os.cpu_count()
    n_splits = 4
    max_shap_data_num = 500
    num_simulation = 1000
    outlier_process = True
    lower_limit=0.01
    upper_limit=0.99

# Create Session
ss = SessionState.get(
    is_model_selection = None,
    df_data = None,
    datas = None,
    output = None,     
    shap_source = None,
    shap_value = None,
    feature_names = None,
    feature_importances = None
)

# Load Data
@st.cache()
def load_data(file_path):
    # Load Data
    df = pd.read_csv(file_path)

    # Select Column
    columns = df.columns
    columns = [column for column in columns if is_numeric_dtype(df[column])]
    df = df[columns]
    return df

# Get File path
file_path = st.selectbox('File', glob('data/*.csv'), index=1)

# Load Data
df_data = load_data(file_path)

# Delete Outlier
if CFG.outlier_process:
    df_data = del_outlier(df_data, lower_limit=CFG.lower_limit, upper_limit=CFG.upper_limit)

# Get Column
df_data_columns = list(df_data.columns)

# Select Target
targets = [st.selectbox('Target', df_data_columns)]

# Select Mode
mode = st.selectbox('Mode', ['Regression', 'Classification'])
if mode == 'Classification':
    # Model and Metric
    _model_list = ['lgb_clf', 'xgb_clf', 'rf_clf', 'et_clf']
    _metric = ['auc', 'accuracy']

    # Convert Value
    values = df_data[targets[0]].dropna().values
    min_value, max_value = float(np.min(values)), float(np.max(values))
    cutoff = st.slider(
        'Cut Off',
        min_value=min_value, 
        max_value=max_value,
        value=(max_value+min_value)/2,
        step=(max_value-min_value)/100
    )
    values = df_data[targets[0]].values
    index_0 = np.where(values <= cutoff) 
    index_1 = np.where(values > cutoff)
    values[index_0] = 0
    values[index_1] = 1
    df_data[targets[0]] = values
    st.text(f'Class 0: {len(index_0[0])}, Class 1: {len(index_1[0])}')

elif mode == 'Regression':
    _model_list = ['lgb_reg', 'xgb_reg', 'rf_reg', 'et_reg']
    _metric = ['mae', 'mse', 'rmse', 'r2']    
else:
    raise ValueError

# Select Models
model_list = st.sidebar.multiselect('Model', _model_list, default=_model_list)

# Select Metric
metric = st.sidebar.selectbox('Metric', _metric, index=0)

# Set Feature Selection
importance_cut_value = st.sidebar.number_input('Importance Cut Value', value=90, min_value=80, max_value=100, step=1)

# Select Feature
st.sidebar.text(f'Select Features')
features = [column for column in df_data_columns if column not in targets]
feature_index = [st.sidebar.checkbox(f'{feature}', value=True) for feature in features]
features = list(np.array(features)[feature_index])

# Button
if st.button(f'Calculate ({targets[0]})'):
    # 1st Training
    st.markdown('### Start Training')
    st.markdown('')
    df_data = select_feature(df_data, features, targets)
    datas = split_data(df_data, features, targets, mode, n_splits=CFG.n_splits)
    output = select_model(model_list, datas, features, targets, metric=metric, n_jobs=CFG.n_jobs)
    shap_source, shap_value = get_shap_value_from_output(output, datas, max_num=CFG.max_shap_data_num)
    feature_names, feature_importances = get_feature_importance(shap_value, features, sort=True)

    # 2nd Training
    if importance_cut_value < 100:
        features = get_important_feature(feature_names, feature_importances, cut=importance_cut_value)
        st.markdown(f'### Feature Selection: {len(feature_names)} > {len(features)}')
        st.text(f'')

        df_data = select_feature(df_data, features, targets)
        datas = split_data(df_data, features, targets, mode, n_splits=CFG.n_splits)
        output = select_model(model_list, datas, features, targets, metric=metric, n_jobs=CFG.n_jobs)
        shap_source, shap_value = get_shap_value_from_output(output, datas, max_num=CFG.max_shap_data_num)
        feature_names, feature_importances = get_feature_importance(shap_value, features, sort=True)

    # Save output to Session
    ss.is_model_selection = True
    ss.df_data = df_data
    ss.datas = datas  
    ss.output = output      
    ss.shap_source = shap_source
    ss.shap_value = shap_value
    ss.feature_names = feature_names
    ss.feature_importances = feature_importances
    gc.collect()

# Plot Graph
if ss.is_model_selection:
    # Load Data
    df_data = ss.df_data
    datas = ss.datas  
    output = ss.output      
    shap_source = ss.shap_source
    shap_value = ss.shap_value
    feature_names = ss.feature_names
    feature_importances = ss.feature_importances

    number = output['fold']
    model_name = output['model_name']
    features = output['features']
    targets = output['targets']
    models = output['models']
    true = output['oob_true']
    pred = output['oob_pred']

    # Plot Training Result
    if mode == 'Regression':
        minimum = np.minimum(np.min(true), np.min(pred))
        maximum = np.maximum(np.max(true), np.max(pred))
        fig, ax = plt.subplots()
        ax.set_title(f'Model: {output["model_name"]}, R$^{2}$: {r2_score(true, pred):.4f}')
        ax.scatter(true, pred, color='#EA4A54')
        ax.plot([minimum, maximum], [minimum, maximum], ls='--', lw=2, color='k')
        ax.set_xlim(minimum, maximum)
        ax.set_ylim(minimum, maximum)
        ax.set_xlabel('Ground Truth', fontsize=13)
        ax.set_ylabel('Prediction', fontsize=13)
        ax.tick_params(axis='both', labelsize=12)
        st.pyplot(fig)

    elif mode == 'Classification':
        st.markdown(f'Accuracy: {100*accuracy_score(true, pred):.2f} %')
        st.markdown(f'AUC: {auc_score(true, pred):.4f}')

    # Plot Feature Importance
    num_init = np.minimum(10, len(feature_names))
    show_number = st.number_input(
        'Feature Number',
        value=num_init,
        min_value=1,
        max_value=len(feature_names),
        step=1
    )
    fig, ax = plt.subplots()
    ax.set_title(f'Target = {targets[0]}')
    ax.barh(feature_names[:show_number][::-1], feature_importances[:show_number][::-1], color='#EA4A54')
    ax.set_xlim(left=0)
    ax.set_ylim(-1, show_number)
    ax.set_xlabel('Feature Importance (%)', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    st.pyplot(fig)

    # Graph and Feature Selection
    graph = st.selectbox('Graph', ['SHAP', 'Simulation'], index=0)
    feature = st.selectbox('Feature', feature_names, index=0)
    index = np.where(np.array(features)==feature)[0][0]

    if graph == 'SHAP':        
        x = shap_source[features[index]].values
        y = shap_value[:, index]
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=20, color='#EA4A54')
        ax.set_title(f'Target = {targets[0]}')
        ax, bottom = add_histogram(ax, x, y)
        ax.set_ylim(bottom=bottom)
        ax.set_xlabel(f'{feature}', fontsize=13)
        ax.set_ylabel(f'SHAP Values for\n{feature}', fontsize=13)
        ax.tick_params(axis='both', labelsize=12)
        st.pyplot(fig)

    elif graph == 'Simulation':
        data = df_data[features[index]].dropna().values
        x_min, x_max = float(np.min(data)), float(np.max(data))
        x_set_min, x_set_max = st.slider(
            'Select a range of values',
            min_value=x_min, 
            max_value=x_max,
            value=(x_min, x_max),
            step=(x_max-x_min)/100
        )
        x, y = simulator(
            df_data, 
            models,
            features, 
            feature, 
            x_set_min, 
            x_set_max, 
            mode=mode,
            n=CFG.num_simulation
        )
        fig, ax = plt.subplots()
        ax.set_title(f'Target = {targets[0]}')
        ax.plot(x, y, color='#EA4A54')
        ax, bottom = add_histogram(ax, data, y)
        ax.set_xlim(x_set_min, x_set_max)
        ax.set_ylim(bottom=bottom)
        ax.set_xlabel(f'{feature}', fontsize=13)
        ax.set_ylabel(f'Pred. {targets[0]}', fontsize=13)
        ax.tick_params(axis='both', labelsize=12)
        st.pyplot(fig)