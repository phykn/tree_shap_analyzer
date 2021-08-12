# Import Library
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import gc
import warnings
warnings.filterwarnings(action='ignore')
from glob import glob
from datetime import datetime
from pandas.api.types import is_numeric_dtype

# Import Module
import module.SessionState as SessionState
from module.metrics import *
from module.load_data import load_data, del_outlier
from module.config import config_from_yaml
from module.split_data import split_data
from module.trainer import select_model
from module.explainer import get_shap_value_from_output, get_feature_importance 
from module.simulation import simulator_1d, simulator_2d
from module.helper import get_df_filter, get_table_download_link
from module.graph import plot_training_result, plot_confusion_matrix, plot_feature_importance, plot_shap, plot_1d_simulation, plot_2d_simulation

# Head
st.markdown('# Tree Model XAI')

# Config
CFG = config_from_yaml('config.yaml')

# Create Session
ss = SessionState.get(is_trained          = None,
                      df_data             = None,
                      datas               = None,
                      output              = None,     
                      shap_source         = None,
                      shap_value          = None,
                      feature_names       = None,
                      feature_importances = None)

# Option
st.sidebar.text('Option')
outlier_process = st.sidebar.checkbox('Delete Target Outlier', value=False)     # Del outlier
backward_elimination = st.sidebar.checkbox('Backward Elimination', value=False) # Backward elimination

# Get File path
file_path = st.selectbox('File Selection', glob('data/*.csv'), index=0)

# Load Data
df_data = load_data(file_path, 
                    max_len           = CFG.DATA.MAX_DATA_NUMBER, 
                    random_state      = CFG.BASE.RANDOM_STATE, 
                    add_random_column = CFG.DATA.ADD_RANDOM_COLUMN)

# Get Column
df_data_columns = list(df_data.columns)

# Filtering
filter_num = st.sidebar.number_input('Filter', 
                                     value     = 0, 
                                     min_value = 0,
                                     max_value = len(df_data_columns), 
                                     step      = 1)
if filter_num > 0:
    for i in range(filter_num):
        df_data = get_df_filter(df_data, 
                                name_1 = f'Filtered column #{i+1}', 
                                name_2 = f'Filtered value #{i+1}')

# Select Mode
mode = st.selectbox('Mode', ['Regression', 'Classification'])

# Select Target
targets = [st.selectbox('Target', df_data_columns)]

# Drop NaN Target
index   = df_data[targets[0]].dropna().index
df_data = df_data.loc[index].reset_index(drop=True)

# Del Outlier
df_data = del_outlier(df_data, targets,
                      lower_limit = CFG.DATA.OUTLIER_LOWER_LIMIT, 
                      upper_limit = CFG.DATA.OUTLIER_UPPER_LIMIT) if outlier_process else df_data

# Apply mode
if mode == 'Classification':
    _model_list = ['lgb_clf', 'xgb_clf', 'rf_clf', 'et_clf']
    _metric = ['ACCURACY', 'AUC', 'LOGLOSS']
    
    # Convert Value
    values = df_data[targets[0]].values
    min_value, max_value = float(np.min(values)), float(np.max(values))
    cutoff = st.slider('Cut Off',
                       min_value = min_value, 
                       max_value = max_value,
                       value     = (max_value+min_value)/2,
                       step      = (max_value-min_value)/100)
    values = np.where(values > cutoff, 1, 0)
    df_data[targets[0]] = values
    
    class0, class1 = st.beta_columns(2)
    with class0:
        st.info(f'Class 0 = {len(values) - np.sum(values)}')
    with class1:
        st.info(f'Class 1 = {np.sum(values)}')

elif mode == 'Regression':
    _model_list = ['lgb_reg', 'xgb_reg', 'rf_reg', 'et_reg']
    _metric = ['R2', 'MAE', 'MSE', 'RMSE']    
else:
    raise ValueError

# Select Models
model_list = st.sidebar.multiselect('Model', _model_list, default=_model_list)

# Select Metric
metric = st.sidebar.selectbox('Metric', _metric, index=0)

# Select Feature
data_qualities = df_data.notnull().sum() / len(df_data)
st.sidebar.text(f'Inputs (data quality | name)')
features = [column for column in df_data_columns if column not in targets]
index = [st.sidebar.checkbox(f'{data_qualities[column]:.2f} | {column}', value=True) for column in features]
features = list(np.array(features)[index])

# Button
if st.button(f'Run ({targets[0]})'):
    print(f'RUN | {file_path} | {datetime.now()}')
    
    # Training
    st.markdown(f'### Training: {len(features)} features')
    datas = split_data(df_data, 
                       features, 
                       targets, 
                       mode, 
                       n_splits     = CFG.TRAIN.N_SPLITS, 
                       shuffle      = True, 
                       random_state = CFG.BASE.RANDOM_STATE)
    output = select_model(model_list, 
                          datas, 
                          features, 
                          targets, 
                          metric       = metric, 
                          random_state = CFG.BASE.RANDOM_STATE, 
                          n_jobs       = CFG.BASE.N_JOBS)
    shap_source, shap_value = get_shap_value_from_output(datas, 
                                                         output,
                                                         max_num      = CFG.GRAPH.SHAP_DATA_NUMBER,
                                                         random_state = CFG.BASE.RANDOM_STATE)
    feature_names, feature_importances = get_feature_importance(shap_value, sort=True)

    # Backward elimination
    if backward_elimination and ('random_noise' in feature_names):
        index = np.where(np.array(feature_names)=='random_noise')[0][0]
        features = feature_names[:index+1]
        st.markdown(f'### Backward Elimination: {len(features)} features')
        st.text('')

        datas = split_data(df_data, 
                           features, 
                           targets, 
                           mode, 
                           n_splits     = CFG.TRAIN.N_SPLITS, 
                           shuffle      = True, 
                           random_state = CFG.BASE.RANDOM_STATE)
        output = select_model(model_list, 
                              datas, 
                              features, 
                              targets, 
                              metric       = metric, 
                              random_state = CFG.BASE.RANDOM_STATE, 
                              n_jobs       = CFG.BASE.N_JOBS)
        shap_source, shap_value = get_shap_value_from_output(datas, 
                                                             output,
                                                             max_num      = CFG.GRAPH.SHAP_DATA_NUMBER,
                                                             random_state = CFG.BASE.RANDOM_STATE)
        feature_names, feature_importances = get_feature_importance(shap_value, sort=True)

    # Save output to Session
    ss.is_trained          = True
    ss.df_data             = df_data
    ss.datas               = datas  
    ss.output              = output      
    ss.shap_source         = shap_source
    ss.shap_value          = shap_value
    ss.feature_names       = feature_names
    ss.feature_importances = feature_importances
    gc.collect()

if ss.is_trained:
    # Load Data
    df_data             = ss.df_data
    datas               = ss.datas  
    output              = ss.output      
    shap_source         = ss.shap_source
    shap_value          = ss.shap_value
    feature_names       = ss.feature_names
    feature_importances = ss.feature_importances

    # Load Train output
    number     = output['fold']
    model_name = output['model_name']
    features   = output['features']
    targets    = output['targets']
    models     = output['models']
    true       = output['oob_true']
    pred       = output['oob_pred']

    # Plot Training Result
    if mode == output['mode']:
        st.markdown('### Result')     

        # Result
        st.markdown(f'Data Number: {len(true)}')
        st.markdown(f'Mode: {mode}')
        st.markdown(f'Model: {model_name}')  
        if mode == 'Regression': 
            st.markdown(f'MAE: {mae_score(true, pred)}')
            st.markdown(f'MSE: {mse_score(true, pred)}')
            st.markdown(f'RMSE: {rmse_score(true, pred)}')
            st.markdown(f'R2 Score: {r2_score(true, pred):.4f}')

            # Graph: Training Result
            st.altair_chart(plot_training_result(true, pred, targets[0]), 
                            use_container_width=True)

        elif mode == 'Classification':
            st.markdown(f'AUC: {auc_score(true, pred)}')
            st.markdown(f'LOGLOSS: {logloss_score(true, pred)}')
            st.markdown(f'Accuracy: {100*accuracy_score(true, pred):.2f} %')

            # Graph: Training Result
            st.pyplot(plot_confusion_matrix(true, pred, targets[0]))

        # Feature importance
        st.markdown('### Feature importance')

        df_feature_importance = pd.DataFrame(
            {'index'     : range(len(feature_names)),
             'feature'   : feature_names,
             'importance': feature_importances}
        )

        # Download Feature Importance        
        href = get_table_download_link(
            df_feature_importance, 
            file_name='importance.csv', 
            title='Download CSV'
        )
        st.markdown(href, unsafe_allow_html=True)

        num_init = np.minimum(10, len(feature_names))
        show_number = st.number_input(
            '표시 개수',
            value=num_init,
            min_value=1,
            max_value=len(feature_names),
            step=1
        )

        # Graph: Feature Importance
        st.altair_chart(
            plot_feature_importance(df_feature_importance, targets[0], num=show_number), 
            use_container_width=True
        )

        # Analysis Type and Feature Selection
        st.markdown('### Explain Model')
        type_name = st.selectbox(
            'Analysis Type (1D Simulation, 2D Simulation and SHAP)', 
            ['1D Simulation', '2D Simulation', 'SHAP'], 
            index=2
        )

        if type_name == 'SHAP':     
            feature = st.selectbox('Feature name', feature_names, index=0)
            x = shap_source[feature].values
            y = shap_value[feature].values

            # Download: 1D Simulation
            href = get_table_download_link(
                pd.DataFrame({feature: x, targets[0]: y}), 
                file_name=f'shap_{feature}_{targets[0]}.csv', 
                title='Download CSV'
            )
            st.markdown(href, unsafe_allow_html=True)

            # Graph: SHAP
            st.altair_chart(plot_shap(x, y, 
                                      shap_source[feature].dropna().values,
                                      feature, targets[0], 
                                      shap_source[targets[0]].mean().round(2)), 
                            use_container_width=True)

        elif type_name == '1D Simulation':
            feature = st.selectbox('Feature name', feature_names, index=0)
            x, y = simulator_1d(df_data, 
                                models,
                                features, 
                                feature, 
                                mode = mode,
                                n    = CFG.GRAPH.SIMULATION_NUMBER)

            # Download: 1D Simulation
            href = get_table_download_link(
                pd.DataFrame({feature: x, targets[0]: y}), 
                file_name=f'1d_simulation_{feature}_{targets[0]}.csv', 
                title='Download CSV'
            )
            st.markdown(href, unsafe_allow_html=True)

            # Graph: 1D Simulation
            st.altair_chart(plot_1d_simulation(x, y, 
                                               df_data[feature].dropna().values,
                                               feature, targets[0]), 
                            use_container_width=True)

        elif type_name == '2D Simulation':
            layout_1, layout_2 = st.beta_columns(2)
            with layout_1:
                feature_1 = st.selectbox('Feature #1', feature_names, index=0)
            with layout_2:
                feature_2 = st.selectbox('Feature #2', feature_names, index=1)

            x1, x2, y = simulator_2d(
                df_data, 
                models,
                features, 
                feature_1, 
                feature_2, 
                mode=mode,
                n=CFG.GRAPH.SIMULATION_NUMBER
            )

            # Download: 2D Simulation
            href = get_table_download_link(
                pd.DataFrame({feature_1: x1, feature_2: x2, targets[0]: y}), 
                file_name=f'2d_simulation_{feature_1}_{feature_2}_{targets[0]}.csv', 
                title='Download CSV'
            )
            st.markdown(href, unsafe_allow_html=True)

            # Graph: Simulation (2D)
            st.altair_chart(plot_2d_simulation(x1, x2, y,
                                               feature_1, feature_2, targets[0]), 
                            use_container_width=True)
