import os
import numpy as np
import pandas as pd
import streamlit as st
import time
from datetime import datetime
from glob import glob
from omegaconf import OmegaConf
from pandas.api.types import is_numeric_dtype
from streamlit_autorefresh import st_autorefresh
from dataloader import read_csv, clear_data
from preprocessing.filter import apply_filter
from preprocessing.target import apply_target, target_encode_numeric, target_encode_category
from preprocessing import delete_nan, replace_nan, delete_outlier, encode_category
from model import split_data, get_best_model
from analysis import get_shap_value, get_importance, simulation_1d, simulation_2d
from graph.evaluation import plot_reg_evaluation, plot_confusion_matrix
from graph.importance import plot_importance
from graph.explanation import plot_shap, plot_simulation_1d, plot_simulation_2d
from graph.matplot import plot_simulation_1d as matplotlib_simulation_1d
from graph.matplot import plot_shap as matplotlib_shap
from helper import get_session_id, encode, convert_figs2zip


# Warning
import warnings
warnings.filterwarnings('ignore')


# # Korean
# import matplotlib
# from matplotlib import font_manager, rc
# font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
# rc('font', family=font_name)
# matplotlib.rcParams['axes.unicode_minus'] = False


# Create Session
if 'config' not in st.session_state:
    st.session_state['config'] = OmegaConf.load('config.yaml')
if 'files' not in st.session_state:
    st.session_state['files'] = np.sort(glob(
        os.path.join(
            st.session_state['config']['file']['root'],
            '*.csv'
        )
    ))
if 'train_file_path' not in st.session_state:
    st.session_state['train_file_path'] = None
if 'filter' not in st.session_state:
    st.session_state['filter'] = None
if 'encoder' not in st.session_state:
    st.session_state['encoder'] = None
if 'target' not in st.session_state:
    st.session_state['target'] = None
if 'feature_all' not in st.session_state:
    st.session_state['feature_all'] = None
if 'feature_selected' not in st.session_state:
    st.session_state['feature_selected'] = None
if 'data_quality' not in st.session_state:
    st.session_state['data_quality'] = None
if 'mode' not in st.session_state:
    st.session_state['mode'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None

if 'state_0' not in st.session_state:
    st.session_state['state_0'] = None
if '_df_0' not in st.session_state:
    st.session_state['_df_0'] = None
if 'state_1' not in st.session_state:
    st.session_state['state_1'] = None
if '_df_1' not in st.session_state:
    st.session_state['_df_1'] = None
if 'state_2' not in st.session_state:
    st.session_state['state_2'] = None
if '_df_2' not in st.session_state:
    st.session_state['_df_2'] = None
if 'state_3' not in st.session_state:
    st.session_state['state_3'] = None
if '_df_3' not in st.session_state:
    st.session_state['_df_3'] = None


# Title
st.markdown('# XAI for tree models')
st.write(f'SESSION ID: {get_session_id()}')


# STEP 1.
st.markdown('### STEP 1. Data preparation')

# Start Time
start_time = time.time()


# State 0: _df_0
state_0 = {}

# Select Train
train_file_path = st.selectbox(
    label = 'Train Data', 
    options = st.session_state['files'],
    index = 0
)
state_0['train_file_path'] = train_file_path

# update _df_0
if (
    state_0 != st.session_state['state_0']
):
    df = read_csv(
        path = state_0['train_file_path'],
        max_len = st.session_state['config']['data']['max_len'],
        add_random_noise = st.session_state['config']['data']['add_random_noise'],
        random_state = st.session_state['config']['setup']['random_state'],
    )
    df = clear_data(df)

    # Update session state
    st.session_state['train_file_path'] = state_0['train_file_path']
    st.session_state['_df_0'] = df
    st.session_state['model'] = None


# Print Options
st.sidebar.write('Options')


# State 1: _df_1
state_1 = {}

# Get Filter Number
num_filter = st.sidebar.number_input(
    label = 'Filter',
    value = 0, 
    min_value = 0,
    max_value = len(st.session_state['_df_0'].columns), 
    step=1
)

# Get Filter Value
filter = {}
if num_filter > 0:
    for i in range(num_filter):
        column = st.selectbox(
            label = f'Filtered column #{i+1}', 
            options = [None]+list(st.session_state['_df_0'].columns),
        )
        if column is not None:
            values = list(
                np.sort(st.session_state['_df_0'][column].dropna().unique())
            )
            selected_values = st.multiselect(
                label = f'Select values #{i+1}',
                options = values, 
                default = values
            )
            filter[column] = selected_values
state_1['filter'] = filter

# Get Mode
mode = st.selectbox(
    label = 'Type', 
    options = ['Regression', 'Binary Classification']
)
state_1['mode'] = mode

# Get Target
target = st.selectbox(
    label = 'Target', 
    options = list(st.session_state['_df_0'].columns)
)
state_1['target'] = target

# Target Encoding
if mode == 'Binary Classification':
    values = st.session_state['_df_0'][target].dropna()
    if is_numeric_dtype(values):
        column_c0, column_i0, column_c1, column_i1 = st.columns(4)

        with column_c0:
            l_q = st.number_input(
                label = 'Label 0 Upper Limit (%)', 
                value = 20, 
                min_value = 0, 
                max_value = 100, 
                step = 1
            )
            state_1['l_q'] = l_q

        with column_c1:
            h_q = st.number_input(
                label = 'Label 0 Lower Limit (%)', 
                value = 80, 
                min_value = 0, 
                max_value = 100, 
                step = 1
            )
            state_1['h_q'] = h_q

        with column_i0:
            st.metric(
                label = 'Label 0 Maximum', 
                value = f"{np.percentile(values, q=l_q):.4f}"
            )

        with column_i1:
            st.metric(
                label = 'Label 1 Minimum', 
                value = f"{np.percentile(values, q=h_q):.4f}"
            )

    else:
        uniques = list(np.sort(np.unique(values)))
        col_0, col_1 = st.columns(2)

        with col_0:
            label_0 = st.selectbox(
                label = 'Label 0', 
                options = uniques,
                index = 0
            )
            state_1['label_0'] = label_0

        with col_1:
            label_1 = st.selectbox(
                label = 'Label 1', 
                options = [column for column in uniques if column != label_0],
                index = 0
            )
            state_1['label_1'] = label_1

# update _df_1
if (
    state_0 != st.session_state['state_0'] or 
    state_1 != st.session_state['state_1']
):
    # Get DF
    df = st.session_state['_df_0'].copy()
    
    # Apply Filter
    df = apply_filter(
        df = df,
        filter = filter
    )

    # Apply Target
    df = apply_target(
        df = df,
        target = target
    )

    # Encode target if the mode is binary classification
    if state_1['mode'] == 'Binary Classification':
        if ('l_q' in state_1) and ('h_q' in state_1):
            df = target_encode_numeric(
                df = df,
                target = state_1['target'],
                l_q = state_1['l_q'],
                h_q = state_1['h_q']
            )
        elif ('label_0' in state_1) and ('label_1' in state_1):
            df = target_encode_category(
                df = df,
                target = state_1['target'],
                label_0 = state_1['label_0'],
                label_1 = state_1['label_1']
            )

    # Update session state    
    st.session_state['filter'] = state_1['filter']
    st.session_state['target'] = state_1['target']
    st.session_state['feature_all'] = [column for column in df.columns if column != state_1['target']]
    st.session_state['data_quality'] = df.notnull().sum() / len(df)
    st.session_state['mode'] = state_1['mode']
    if ('l_q' in state_1) and ('h_q' in state_1):
        st.session_state['l_q'] = state_1['l_q']
        st.session_state['h_q'] = state_1['h_q']
        st.session_state['label_0'] = None
        st.session_state['label_1'] = None
    elif ('label_0' in state_1) and ('label_1' in state_1):
        st.session_state['l_q'] = None
        st.session_state['h_q'] = None
        st.session_state['label_0'] = state_1['label_0']
        st.session_state['label_1'] = state_1['label_1']
    else:
        st.session_state['l_q'] = None
        st.session_state['h_q'] = None
        st.session_state['label_0'] = None
        st.session_state['label_1'] = None
    st.session_state['_df_1'] = df
    st.session_state['model'] = None


# State 2: _df_2
state_2 = {}

# NaN Data
nan_data = st.sidebar.selectbox(
    label = 'NaN Data',
    options = ['Delete', 'Replace']
)
state_2['nan_data'] = nan_data

# Auto Feature Selection
auto_feature_selection = st.sidebar.selectbox(
    label = 'Auto Feature Selection',
    options = [False, True]
)
state_2['auto_feature_selection'] = auto_feature_selection

# update _df_2
if (
    state_0 != st.session_state['state_0'] or 
    state_1 != st.session_state['state_1'] or
    state_2 != st.session_state['state_2']
):
    # Get DF
    df = st.session_state['_df_1'].copy()

    # Encode Data
    df, encoder = encode_category(df)

    # Update session state    
    st.session_state['nan_data'] = state_2['nan_data']
    st.session_state['auto_feature_selection'] = auto_feature_selection
    st.session_state['encoder'] = encoder
    st.session_state['_df_2'] = df.reset_index(drop=True)
    st.session_state['model'] = None


# State 3: _df_3
state_3 = {}

# Select Features
st.sidebar.markdown("""---""")
st.sidebar.write('Features')
st.sidebar.text(f'Data quality | name')
index = [
    st.sidebar.checkbox(
        label = f"{st.session_state['data_quality'][column]:.2f} | {column}",
        key = f"_{column}",
        value = True, 
    ) for column in st.session_state['feature_all']
]
feature_selected = list(np.array(st.session_state['feature_all'])[index])
state_3['feature_selected'] = feature_selected

# Magage Features
def uncheck():
    for column in st.session_state['feature_all']:
        st.session_state[f'_{column}'] = False
def check():
    for column in st.session_state['feature_all']:
        st.session_state[f'_{column}'] = True
        
_, col_1, col_2 = st.sidebar.columns([1, 4, 5])
with col_1:
    st.button(
        label = 'Check All', 
        on_click = check
    )  
with col_2:
    st.button(
        label = 'Uncheck All', 
        on_click = uncheck
    )

# update _df_3
if (
    state_0 != st.session_state['state_0'] or 
    state_1 != st.session_state['state_1'] or
    state_2 != st.session_state['state_2'] or
    state_3 != st.session_state['state_3']
):
    # Get DF
    df = st.session_state['_df_2'].copy()

    # Select columns
    columns = state_3['feature_selected'] + [st.session_state['target']]
    df = df[columns]

    # Update session state    
    st.session_state['feature_selected'] = state_3['feature_selected']
    st.session_state['_df_3'] = df
    st.session_state['model'] = None


# Update states
st.session_state['state_0'] = state_0
st.session_state['state_1'] = state_1
st.session_state['state_2'] = state_2
st.session_state['state_3'] = state_3


# Data wall time
wall_time = time.time() - start_time


# Print Information
st.sidebar.markdown("""---""")
st.sidebar.write(f"Wall time: {wall_time:.4f} sec")
st.sidebar.write(f"Data Num: {len(st.session_state['_df_3'])}")
st.sidebar.write(f"Target: {st.session_state['target']}")
st.sidebar.write(f"Feature Num: {len(feature_selected)}")


# Print Encoder
columns = st.session_state['feature_selected'] + [st.session_state['target']]
encoder = {}
if len(st.session_state['encoder']) > 0:    
    for column in columns:
        if column in st.session_state['encoder']:
            encoder[column] = st.session_state['encoder'][column]

if len(encoder) > 0:
    st.sidebar.write('Encoded Features')
    st.sidebar.write(encoder)


# Print DF
st.write('Sample Data (5)')
st.write(st.session_state['_df_3'].iloc[:5])


# Train Model
if st.session_state['model'] is None:
    st.markdown("""---""")
    if st.button('Start Model Training'):
        # Log
        time_now = str(datetime.now())[:19]
        print(f'START | {time_now} | {get_session_id()} | {st.session_state["train_file_path"]}')

        # Load Data
        df = st.session_state['_df_3'].copy()
        features = st.session_state['feature_selected']
        target = st.session_state['target']
        if st.session_state['mode'] == 'Regression':
            mode = 'reg'
        if st.session_state['mode'] == 'Binary Classification':
            mode = 'clf'

        # NaN Data
        df = df[features+[target]].copy()

        if df.isna().sum().sum() == 0:
            st.session_state['nan_processed'] = False
        else:
            if st.session_state['nan_data'] == 'Delete':
                df = delete_nan(df)
            elif st.session_state['nan_data'] == 'Replace':
                df = replace_nan(
                    df = df, 
                    random_state = st.session_state['config']['setup']['random_state']
                )
            st.session_state['nan_processed'] = True

        st.session_state['data_num'] = len(df)

        # Dataset
        datasets = split_data(
            df = df,
            features = features,
            target = target,
            mode = mode,
            n_splits = st.session_state['config']['split']['n_splits'],
            shuffle = True,
            random_state = st.session_state['config']['setup']['random_state']
        )

        # Best Model
        best_model, history = get_best_model(
            datasets = datasets, 
            mode = mode,
            random_state = st.session_state['config']['setup']['random_state'],
            n_jobs = st.session_state['config']['setup']['n_jobs']
        )
        best_model['features'] = features
        best_model['target'] = target
        best_model['datasets'] = datasets

        # SHAP
        source, shap_value = get_shap_value(
            config = best_model,
            max_num = st.session_state['config']['shap']['max_num']
        )
        output = get_importance(
            shap_value,
            sort = st.session_state['config']['importance']['sort'],
            normalize = st.session_state['config']['importance']['normalize']
        )

        shap = {}
        shap['features'] = output['features']
        shap['importance'] = output['importance']
        shap['source'] = source
        shap['shap_value'] = shap_value

        if (
            st.session_state['auto_feature_selection'] and
            'random_noise' in shap['features']
        ):
            features = shap['features']
            index = np.where(np.array(features)=='random_noise')[0][0]
            if index != 0:
                # Print Info
                st.write('Auto Feature Selection is ON.')

                # Set new features
                features = features[:index]

                # Dataset
                datasets = split_data(
                    df = df,
                    features = features,
                    target = target,
                    mode = mode,
                    n_splits = st.session_state['config']['split']['n_splits'],
                    shuffle = True,
                    random_state = st.session_state['config']['setup']['random_state']
                )

                # Best Model
                best_model, history = get_best_model(
                    datasets = datasets, 
                    mode = mode,
                    random_state = st.session_state['config']['setup']['random_state'],
                    n_jobs = st.session_state['config']['setup']['n_jobs']
                )
                best_model['features'] = features
                best_model['target'] = target
                best_model['datasets'] = datasets

                # SHAP
                source, shap_value = get_shap_value(
                    config = best_model,
                    max_num = st.session_state['config']['shap']['max_num']
                )
                output = get_importance(
                    shap_value,
                    sort = st.session_state['config']['importance']['sort'],
                    normalize = st.session_state['config']['importance']['normalize']
                )

                shap = {}
                shap['features'] = output['features']
                shap['importance'] = output['importance']
                shap['source'] = source
                shap['shap_value'] = shap_value 


        # Update session state
        st.session_state['history'] = history
        st.session_state['model'] = best_model
        st.session_state['shap'] = shap

        # Refresh page
        st_autorefresh(interval=100, limit=2)


# Result
else:
    # STEP 2. Evaluation
    st.markdown('### STEP 2. Evaluation')

    # NaN Data
    if st.session_state['nan_processed']:
        st.write(f"NaN Data process mode is {st.session_state['nan_data']}.")

    # Data number
    st.write(f"Data Number: {st.session_state['data_num']}")

    # Print Best Model
    best = {}
    best['name'] = st.session_state['model']['name']
    best.update(st.session_state['model']['score'])
    st.write('Best Model')
    st.write(best)

    # Print Score
    st.write(st.session_state['history'])

    # Graph
    if st.session_state['mode'] == 'Regression':
        st.altair_chart(
            plot_reg_evaluation(
                true = st.session_state['model']['oob_true'],
                pred = st.session_state['model']['oob_pred'],
                target = st.session_state['model']['target']
            ), 
            use_container_width = True
        )
    elif st.session_state['mode'] == 'Binary Classification':
        st.pyplot(
            plot_confusion_matrix(
                true = st.session_state['model']['oob_true'],
                pred = st.session_state['model']['oob_pred'],
                target = st.session_state['model']['target']
            )
        )


    # STEP 3. Feature Importance
    features = st.session_state['shap']['features']
    importance = st.session_state['shap']['importance']

    col_1, col_2 = st.columns([3, 1])
    with col_1:
        st.markdown('### STEP 3. Feature Importance')
    with col_2:
        show_number = st.number_input(
            label = 'Number',
            value = np.minimum(10, len(features)),
            min_value = 1,
            max_value = len(features),
            step = 1
        )

    st.altair_chart(
        plot_importance(
            features = features,
            importance = importance,
            target = st.session_state['model']['target'], 
            num = show_number
        ), 
        use_container_width=True
    )

    # Download CSV
    df_importance = pd.DataFrame()
    df_importance['feature'] = features
    df_importance['importance'] = importance
    st.download_button(
        label = 'Download (.csv)',
        data = df_importance.to_csv(index=False).encode('utf-8-sig'),
        file_name = f'importance.csv',
        mime = 'text/csv'
    )


    # STEP 4. Local Explanation
    df = df = st.session_state['_df_3']
    source = st.session_state['shap']['source']
    shap_value = st.session_state['shap']['shap_value']

    col_1, col_2 = st.columns([3, 1])
    with col_1:
        st.markdown('### STEP 4. Local Explanation')
    with col_2:
        type_name = st.selectbox(
            label = 'Type',
            options = ['SHAP', '1D Simulation', '2D Simulation']
        )
    if type_name == 'SHAP':
        feature = st.selectbox(
            label = 'Feature', 
            options = features
        )
        st.altair_chart(
            plot_shap(
                x = source[feature].values, 
                y = shap_value[feature].values, 
                x_all = df[feature].dropna().values,
                feature = feature,
                target = st.session_state['model']['target'],
                mean = np.mean(st.session_state['model']['oob_true'])
            ),
            use_container_width = True
        )   

        # Print Encode
        if feature in st.session_state['encoder']:
            st.write(feature)
            st.write(st.session_state['encoder'][feature])

        # Download CSV
        df_shap = pd.DataFrame()
        df_shap[feature] = source[feature].values
        df_shap['SHAP Value'] = shap_value[feature].values
        st.download_button(
            label = 'Download (.csv)',
            data = df_shap.to_csv(index=False).encode('utf-8-sig'),
            file_name = f'shap.csv',
            mime = 'text/csv'
        )

        # Download figures
        col_0, col_1 = st.columns(2)
        with col_0:
            if st.button('Extract all figures'):
                progress = st.progress(0)
                figs = []
                for i, feature in enumerate(features):
                    # get figure
                    figs.append(
                        matplotlib_shap(
                            x = source[feature].values, 
                            y = shap_value[feature].values, 
                            x_all = df[feature].dropna().values,
                            feature = feature,
                            target = st.session_state['model']['target'],
                            mean = np.mean(st.session_state['model']['oob_true'])
                        )
                    )
                    # Update progress
                    progress.progress((i+1)/len(features))

                # convert to zip
                with col_1:
                    st.download_button(
                        label = 'Download (.zip)',
                        data = convert_figs2zip(figs),
                        file_name = 'shap.zip',
                        mime="application/octet-stream"
                    )

    elif type_name == '1D Simulation':
        feature = st.selectbox(
            label = 'Feature', 
            options = features
        )
        x, y = simulation_1d(
            datasets = st.session_state['model']['datasets'],
            models = st.session_state['model']['models'],
            features = st.session_state['model']['features'],
            feature = feature,
            mode = st.session_state['model']['type'],
            num = st.session_state['config']['simulation']['num']
        )
        st.altair_chart(
            plot_simulation_1d(
                x = x, 
                y = y,
                x_all = df[feature].dropna().values,
                feature = feature, 
                target = st.session_state['model']['target']
            ),
            use_container_width = True
        )

        # Print Encode
        if feature in st.session_state['encoder']:
            st.write(feature)
            st.write(st.session_state['encoder'][feature])

        # Download CSV
        df_1d = pd.DataFrame()
        df_1d[feature] = x
        df_1d['Prediction'] = y
        st.download_button(
            label = 'Download (.csv)',
            data = df_1d.to_csv(index=False).encode('utf-8-sig'),
            file_name = f'1d_simulation.csv',
            mime = 'text/csv'
        )

        # Download figures
        col_0, col_1 = st.columns(2)
        with col_0:
            if st.button('Extract all figures'):
                progress = st.progress(0)
                figs = []
                for i, feature in enumerate(features):
                    # get x and y
                    x, y = simulation_1d(
                        datasets = st.session_state['model']['datasets'],
                        models = st.session_state['model']['models'],
                        features = st.session_state['model']['features'],
                        feature = feature,
                        mode = st.session_state['model']['type'],
                        num = st.session_state['config']['simulation']['num']
                    )
                    # get figure
                    figs.append(
                        matplotlib_simulation_1d(
                            x = x, 
                            y = y,
                            x_all = df[feature].dropna().values,
                            feature = feature, 
                            target = st.session_state['model']['target']
                        )
                    )
                    # Update progress
                    progress.progress((i+1)/len(features))

                # convert to zip
                with col_1:
                    st.download_button(
                        label = 'Download (.zip)',
                        data = convert_figs2zip(figs),
                        file_name = '1d_simulation.zip',
                        mime="application/octet-stream"
                    )


    elif type_name == '2D Simulation':
        col_0, col_1 = st.columns(2)
        with col_0:
            feature_0 = st.selectbox(
                label = 'Feature #1', 
                options = features
            )
        with col_1:
            feature_1 = st.selectbox(
                label = 'Feature #2', 
                options = [feature for feature in features if feature != feature_0]
            )

        x_0, x_1, y = simulation_2d(
            datasets = st.session_state['model']['datasets'],
            models = st.session_state['model']['models'],
            features = st.session_state['model']['features'],
            feature_0 = feature_0,
            feature_1 = feature_1,
            mode = st.session_state['model']['type'],
            num = st.session_state['config']['simulation']['num']
        )

        st.altair_chart(
            plot_simulation_2d(
                x_0 = x_0,
                x_1 = x_1,
                y = y,
                feature_0 = feature_0,
                feature_1 = feature_1, 
                target = st.session_state['model']['target']
            ),
            use_container_width = True
        )

        # Print Encode
        if feature_0 in st.session_state['encoder']:
            st.write(feature_0)
            st.write(st.session_state['encoder'][feature_0])

        if feature_1 in st.session_state['encoder']:
            st.write(feature_1)
            st.write(st.session_state['encoder'][feature_1])

        # Download CSV
        df_2d = pd.DataFrame()
        df_2d[feature_0] = x_0
        df_2d[feature_1] = x_1
        df_2d['Prediction'] = y
        st.download_button(
            label = 'Download (.csv)',
            data = df_2d.to_csv(index=False).encode('utf-8-sig'),
            file_name = f'2d_simulation.csv',
            mime = 'text/csv'
        )


    # STEP 5. Prediction
    st.markdown('### STEP 5. Prediction')

    test_file_path = st.selectbox(
        label = 'Test Data',
        options = st.session_state['files'],
        index = 1
    )
    col_0, col_1 = st.columns(2)
    with col_0:
        if st.button('Generate Prediction file'):
            # Load Data
            df_test = read_csv(
                path = test_file_path,
                max_len = None,
                add_random_noise = st.session_state['config']['data']['add_random_noise'],
                random_state = st.session_state['config']['setup']['random_state'],
            )

            # Check features
            features = st.session_state['model']['features']
            n_features = []       
            for feature in features:
                if feature not in list(df_test.columns):
                    n_features.append(feature)

            if len(n_features) != 0:
                st.write(f"TEST File: {test_file_path}")
                st.write(f"{n_features} are not in the test file.")
            else:
                with st.spinner(text="In progress..."):
                    # Apply filter
                    df_test = apply_filter(
                        df = df_test,
                        filter = st.session_state['filter']
                    )

                    # Fill NaN Data
                    df_data = replace_nan(
                        df = df_test.copy(), 
                        random_state = st.session_state['config']['setup']['random_state']
                    )

                    # Encode Data
                    df_data = encode(
                        df = df_data,
                        encoder = st.session_state['encoder']
                    )

                    # Prediction
                    inputs = df_data[features].values
                    pred = []
                    for model in st.session_state['model']['models']:
                        if st.session_state['model']['type'] == 'reg':
                            p = model.predict(inputs)
                        elif st.session_state['model']['type'] == 'clf':
                            p = model.predict_proba(inputs)[:, 1]
                        pred.append(p)
                    pred = np.mean(pred, axis=0)

                    # Output Data
                    target = st.session_state['model']['target']
                    df_test[f'pred_{target}'] = pred
                    if 'random_noise' in df_test.columns:
                        df_test = df_test.drop(columns=['random_noise'])

                # Download CSV
                path = test_file_path
                name = os.path.basename(path) 
           
                with col_1:
                    st.download_button(
                        label = 'Download (.csv)',
                        data = df_test.to_csv(index=False).encode('utf-8-sig'),
                        file_name = f'pred_{name}',
                        mime = 'text/csv'
                    )
