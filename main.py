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
from module.metrics import *
from module.data_preparation import select_feature, split_data, del_outlier
from module.model_selection import select_model
from module.shap import get_feature_importance, get_shap_value_from_output, get_important_feature
from module.simulation import simulator_1d, simulator_2d
from module.helper import add_histogram
import module.SessionState as SessionState
import warnings
warnings.filterwarnings(action='ignore')

# Head
st.markdown('# Model Analyzer ver 0.8')

# Config
class CFG:
    n_jobs = os.cpu_count()
    n_splits = 4
    max_data_length = 100000
    max_shap_data_num = 500
    num_simulation = 100
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
def load_data(file_path, max_len=100000):
    # Load Data
    df = pd.read_csv(file_path)
    
    if len(df) > max_len:
        df = df.sample(n=max_len, random_state=42)
        
    # Select Column
    columns = df.columns
    columns = [column for column in columns if is_numeric_dtype(df[column])]
    df = df[columns]
    return df

# Get File path
file_path = st.selectbox('Select a sample file', glob('data/*.csv'), index=0)

# Upload File
# uploaded_file = st.file_uploader('or upload a CSV file', type=['csv'])
# if uploaded_file is not None:
#     file_path = uploaded_file

# Load Data
df_data = load_data(file_path, max_len=CFG.max_data_length)

# Delete Outlier
if CFG.outlier_process:
    df_data = del_outlier(df_data, lower_limit=CFG.lower_limit, upper_limit=CFG.upper_limit)

# Get Column
df_data_columns = list(df_data.columns)

# Select Mode
mode = st.selectbox('Mode', ['Regression', 'Classification'])

# Select Target
targets = [st.selectbox('Target', df_data_columns)]
    
if mode == 'Classification':
    _model_list = ['lgb_clf', 'xgb_clf', 'rf_clf', 'et_clf']
    _metric = ['AUC', 'LOGLOSS', 'ACCURACY']
    
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
    
    class0, class1 = st.beta_columns(2)
    with class0:
        st.info(f'Class 0 = {len(index_0[0])}')
    with class1:
        st.info(f'Class 1 = {len(index_1[0])}')

elif mode == 'Regression':
    _model_list = ['lgb_reg', 'xgb_reg', 'rf_reg', 'et_reg']
    _metric = ['MAE', 'MSE', 'RMSE', 'R2']    
else:
    raise ValueError

# Select Models
model_list = st.sidebar.multiselect('Model', _model_list, default=_model_list)

# Select Metric
metric = st.sidebar.selectbox('Metric', _metric, index=0)

# Set Feature Selection
importance_cut_value = st.sidebar.number_input('Importance Cut Value', value=95, min_value=80, max_value=100, step=1)

# Select Feature
st.sidebar.text(f'Select Features')
features = [column for column in df_data_columns if column not in targets]
feature_index = [st.sidebar.checkbox(f'{feature}', value=True) for feature in features]
features = list(np.array(features)[feature_index])

# Button
if st.button(f'RUN ({targets[0]})'):
    print(f'RUN | {file_path} | {datetime.now()}')
    
    # 1st Training
    st.markdown('### Start Training')
    st.text('')
    df_data = select_feature(df_data, features, targets)
    datas = split_data(df_data, features, targets, mode, n_splits=CFG.n_splits)
    output = select_model(model_list, datas, features, targets, metric=metric, n_jobs=CFG.n_jobs)
    shap_source, shap_value = get_shap_value_from_output(output, datas, max_num=CFG.max_shap_data_num)
    feature_names, feature_importances = get_feature_importance(shap_value, features, sort=True)

    # 2nd Training
    if importance_cut_value < 100:
        features = get_important_feature(feature_names, feature_importances, cut=importance_cut_value)
        st.markdown(f'### Feature Selection: {len(feature_names)} > {len(features)}')
        st.text('')

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
    if mode == output['mode']:
        st.markdown('### Training Result')
        st.markdown(f'Mode: {mode}')
        st.markdown(f'Model: {model_name}')  

        if mode == 'Regression': 
            st.markdown(f'MAE: {mae_score(true, pred)}')
            st.markdown(f'MSE: {mse_score(true, pred)}')
            st.markdown(f'RMSE: {rmse_score(true, pred)}')
            st.markdown(f'R2 Score: {r2_score(true, pred):.4f}')
            st.text('')

            minimum = np.minimum(np.min(true), np.min(pred))
            maximum = np.maximum(np.max(true), np.max(pred))
            fig, ax = plt.subplots()
            ax.set_title(f'Target = {targets[0]}', fontsize=13)
            ax.scatter(true, pred, color='#EA4A54')
            ax.plot([minimum, maximum], [minimum, maximum], ls='--', lw=2, color='k')
            ax.set_xlim(minimum, maximum)
            ax.set_ylim(minimum, maximum)
            ax.set_xlabel('Ground Truth', fontsize=13)
            ax.set_ylabel('Prediction', fontsize=13)
            ax.tick_params(axis='both', labelsize=12)
            st.pyplot(fig)

        elif mode == 'Classification':
            st.markdown(f'AUC: {auc_score(true, pred)}')
            st.markdown(f'LOGLOSS: {logloss_score(true, pred)}')
            st.markdown(f'Accuracy: {100*accuracy_score(true, pred):.2f} %')
            st.text('')

        # Plot Feature Importance
        st.markdown('### Feature Importance')
        num_init = np.minimum(10, len(feature_names))
        show_number = st.number_input(
            'Feature Number',
            value=num_init,
            min_value=1,
            max_value=len(feature_names),
            step=1
        )
        fig, ax = plt.subplots()
        ax.set_title(f'Target = {targets[0]}', fontsize=13)
        ax.barh(feature_names[:show_number][::-1], feature_importances[:show_number][::-1], color='#EA4A54')
        ax.set_xlim(left=0)
        ax.set_ylim(-1, show_number)
        ax.set_xlabel('Feature Importance (%)', fontsize=13)
        ax.tick_params(axis='both', labelsize=12)
        st.pyplot(fig)

        # Analysis Type and Feature Selection
        st.markdown('### Model Analysis')
        type_name = st.selectbox('Type', ['SHAP', 'Simulation (1D)', 'Simulation (2D)'], index=0)

        if type_name == 'SHAP':     
            feature = st.selectbox('Feature', feature_names, index=0)
            index = np.where(np.array(features)==feature)[0][0]   
            x = shap_source[features[index]].values
            y = shap_value[:, index]
            fig, ax = plt.subplots()
            ax.set_title(f'Target = {targets[0]}', fontsize=13)
            ax.scatter(x, y, s=20, color='#EA4A54')        
            ax, bottom = add_histogram(ax, x, y)
            ax.set_ylim(bottom=bottom)
            ax.set_xlabel(f'{feature}', fontsize=13)
            ax.set_ylabel(f'SHAP Values for\n{feature}', fontsize=13)
            ax.tick_params(axis='both', labelsize=12)
            st.pyplot(fig)

        elif type_name == 'Simulation (1D)':
            layout_1, layout_2 = st.beta_columns(2)

            with layout_1:
                feature = st.selectbox('Feature', feature_names, index=0)
            index = np.where(np.array(features)==feature)[0][0]  
            data = df_data[features[index]].dropna().values
            x_min, x_max = float(np.min(data)), float(np.max(data))

            with layout_2:
                x_set_min, x_set_max = st.slider(
                    'Range',
                    min_value=x_min, 
                    max_value=x_max,
                    value=(x_min, x_max),
                    step=(x_max-x_min)/100
                )        

            x, y = simulator_1d(
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
            ax.set_title(f'Target = {targets[0]}', fontsize=13)
            ax.plot(x, y, color='#EA4A54')
            ax, bottom = add_histogram(ax, data, y)
            ax.set_xlim(x_set_min, x_set_max)
            ax.set_ylim(bottom=bottom)
            ax.set_xlabel(f'{feature}', fontsize=13)
            ax.set_ylabel(f'Pred. {targets[0]}', fontsize=13)
            ax.tick_params(axis='both', labelsize=12)
            st.pyplot(fig)

        elif type_name == 'Simulation (2D)':
            layout_1, layout_2 = st.beta_columns(2)
            layout_3, layout_4 = st.beta_columns(2)

            with layout_1:
                feature_1 = st.selectbox('Feature #1', feature_names, index=0)
            index_1 = np.where(np.array(features)==feature_1)[0][0]  
            data_1 = df_data[features[index_1]].dropna().values
            x1_min, x1_max = float(np.min(data_1)), float(np.max(data_1))
            with layout_2:
                x1_set_min, x1_set_max = st.slider(
                    'Range #1',
                    min_value=x1_min, 
                    max_value=x1_max,
                    value=(x1_min, x1_max),
                    step=(x1_max-x1_min)/100
                )     

            with layout_3:
                feature_2 = st.selectbox('Feature #2', feature_names, index=0)
            index_2 = np.where(np.array(features)==feature_2)[0][0]  
            data_2 = df_data[features[index_2]].dropna().values
            x2_min, x2_max = float(np.min(data_2)), float(np.max(data_2))
            with layout_4:
                x2_set_min, x2_set_max = st.slider(
                    'Range #2',
                    min_value=x2_min, 
                    max_value=x2_max,
                    value=(x2_min, x2_max),
                    step=(x2_max-x2_min)/100
                )      

            x1, x2, y = simulator_2d(
                df_data, 
                models,
                features, 
                feature_1, 
                x1_set_min, 
                x1_set_max, 
                feature_2, 
                x2_set_min, 
                x2_set_max, 
                mode=mode,
                n=CFG.num_simulation
            )

            fig, ax = plt.subplots()
            ax.set_title(f'Target = {targets[0]}', fontsize=13)
            c = ax.pcolormesh(
                x1, x2, y, 
                cmap='jet', 
                vmin=np.min(y), 
                vmax=np.max(y)
            )
            ax.set_xlim(x1_set_min, x1_set_max)
            ax.set_ylim(x2_set_min, x2_set_max)
            ax.set_xlabel(f'{feature_1}', fontsize=13)
            ax.set_ylabel(f'{feature_2}', fontsize=13)
            ax.tick_params(axis='both', labelsize=12)
            fig.colorbar(c)
            st.pyplot(fig)
            
    else:
        ss.is_model_selection = None
        ss.df_data = None
        ss.datas = None
        ss.output = None  
        ss.shap_source = None
        ss.shap_value = None
        ss.feature_names = None
        ss.feature_importances = None
