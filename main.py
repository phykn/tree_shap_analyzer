import os
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

from glob import glob
from omegaconf import OmegaConf
from pandas.api.types import is_numeric_dtype
from streamlit_autorefresh import st_autorefresh

from src.dataloader import read_csv, clear_data
from src.model import split_data, get_best_model
from src.helper import get_session_id, encode, convert_figs2zip
from src.preprocessing import *
from src.analysis import *
from src.graph import *


# warning
warnings.filterwarnings("ignore")

# create session
if "config" not in st.session_state:
    st.session_state["config"] = OmegaConf.load("config.yaml")
if "files" not in st.session_state:
    st.session_state["files"] = glob(os.path.join(st.session_state["config"]["file"]["root"], "*.csv"))
if "state" not in st.session_state:
    st.session_state["state"] = {}
if "data" not in st.session_state:
    st.session_state["data"] = {}
if "model" not in st.session_state:
    st.session_state["model"] = None

# title
st.markdown("# Tree shap analyzer")
st.markdown("### 1. Data preparation")

state = {}
start_time = time.time()


############################ STATE 0 START ############################
# csv data file selection / encode categorical features

# state initialize
state["0"] = {}

# select data
train_file_path = st.selectbox("Train data file", st.session_state["files"])
state["0"]["train_file_path"] = train_file_path

# update
if (
    "0" not in st.session_state["state"] or
    state["0"] != st.session_state["state"]["0"]
):
    df = read_csv(
        path=state["0"]["train_file_path"],
        max_len=st.session_state["config"]["data"]["max_len"],
        add_random_noise=st.session_state["config"]["data"]["add_random_noise"],
        random_state=st.session_state["config"]["setup"]["random_state"],
    )
    df = clear_data(df)

    # encode data
    _, encoder = encode_category(df.copy())

    # update session state
    st.session_state["state"]["train_file_path"] = state["0"]["train_file_path"]
    st.session_state["state"]["encoder"] = encoder
    st.session_state["data"]["0"] = df
    st.session_state["model"] = None
############################ STATE 0 END ##############################


############################ STATE 1 START ############################
# target / option selection

# state initialize
state["1"] = {}

# read data
df = st.session_state["data"]["0"]

# select target
target = st.selectbox("Target", df.columns.tolist())
state["1"]["target"] = target

st.sidebar.write("Option")

# auto feature selection
auto_feature_selection = st.sidebar.selectbox("Auto Feature Selection", [False, True])
state["1"]["auto_feature_selection"] = auto_feature_selection

st.sidebar.markdown("---")

st.sidebar.write("Data Preprocessing")

# NaN data
nan_data = st.sidebar.selectbox("Missing value", ["Delete", "Replace"])
state["1"]["nan_data"] = nan_data

# update
if (
    "1" not in st.session_state["state"] or
    state["0"] != st.session_state["state"]["0"] or 
    state["1"] != st.session_state["state"]["1"]
):
    # update session state
    st.session_state["state"]["target"] = state["1"]["target"]
    st.session_state["state"]["nan_data"] = state["1"]["nan_data"]
    st.session_state["state"]["auto_feature_selection"] = state["1"]["auto_feature_selection"]
    st.session_state["state"]["all_features"] = [column for column in df.columns if column != target]
    st.session_state["data"]["1"] = df
    st.session_state["model"] = None
############################ STATE 1 END ##############################


############################ STATE 2 START ############################
# apply filter

# state initialize
state["2"] = {}

# read data
df = st.session_state["data"]["1"]

# get filter number
num_filter = st.sidebar.number_input(
    label="Filter",
    value=0, 
    min_value=0,
    max_value=len(df.columns), 
    step=1
)

# get filter value
filter = {}
if num_filter > 0:
    for i in range(num_filter):
        column = st.selectbox(
            label=f"Filtered column #{i+1}", 
            options=[None]+list(df.columns),
        )
        if column is not None:
            values=np.sort(df[column].dropna().unique())
            if is_numeric_dtype(values):
                selected_value_i, selected_value_f = st.select_slider(
                    label=f"Select values #{i+1}",
                    options=values,
                    value=(np.min(values), np.max(values))
                )
                selected_values = values[np.where(np.logical_and(values>=selected_value_i, values<=selected_value_f))]
                selected_values = selected_values.tolist()
            else:
                selected_values=st.multiselect(
                    label=f"Select values #{i+1}",
                    options=values, 
                    default=values
                )
            filter[column]=selected_values

state["2"]["filter"] = filter

# update
if (
    "2" not in st.session_state["state"] or
    state["0"] != st.session_state["state"]["0"] or
    state["1"] != st.session_state["state"]["1"] or
    state["2"] != st.session_state["state"]["2"]
):    
    # apply filter
    df = apply_filter(df.copy(), filter)

    # update session state
    st.session_state["state"]["filter"] = state["2"]["filter"]
    st.session_state["state"]["data_quality"] = df.notnull().sum() / len(df)
    st.session_state["data"]["2"] = df
    st.session_state["model"] = None
############################ STATE 2 END ##############################


############################ STATE 3 START ############################
# select features

# state initialize
state["3"] = {}

# read data
df = st.session_state["data"]["2"]

# select features
st.sidebar.markdown("""---""")
st.sidebar.write("Features")
st.sidebar.text("Data quality | name")
index = [
    st.sidebar.checkbox(
        label = f"{st.session_state['state']['data_quality'][column]:.2f} | {column}",
        key = f"_{column}",
        value = True, 
    ) for column in st.session_state["state"]["all_features"]
]
selected_features = np.array(st.session_state["state"]["all_features"])[index].tolist()
state["3"]["selected_features"] = selected_features

# manage features
def uncheck():
    for column in st.session_state["state"]["all_features"]:
        st.session_state[f"_{column}"] = False
def check():
    for column in st.session_state["state"]["all_features"]:
        st.session_state[f"_{column}"] = True
        
_, col_1, col_2 = st.sidebar.columns([1, 4, 5])
with col_1:
    st.button("Check All", on_click=check)  
with col_2:
    st.button("Uncheck All", on_click=uncheck)

# update
if (
    "3" not in st.session_state["state"] or
    state["0"] != st.session_state["state"]["0"] or 
    state["1"] != st.session_state["state"]["1"] or
    state["2"] != st.session_state["state"]["2"] or
    state["3"] != st.session_state["state"]["3"]
):
    # select columns
    columns = state["3"]["selected_features"] + [st.session_state["state"]["target"]]
    df = df.copy()[columns]

    # update session state
    st.session_state["state"]["selected_features"] = state["3"]["selected_features"]
    st.session_state["data"]["3"] = df
    st.session_state["model"] = None    
############################ STATE 3 END ##############################


############################ STATE 4 START ############################
# target encoding / nan process

# state initialize
state["4"] = {}

# read data
features = st.session_state["state"]["selected_features"]
target = st.session_state["state"]["target"]
df = st.session_state["data"]["3"]
df = apply_target(df.copy(), target)

# nan data process
df = df[features+[target]].copy()
if df.isna().sum().sum() > 0:
    if st.session_state["state"]["nan_data"] == "Delete":
        df = delete_nan(df)
    elif st.session_state["state"]["nan_data"] == "Replace":
        df = replace_nan(df, random_state=st.session_state["config"]["setup"]["random_state"])

# get mode
mode = st.selectbox("Solver", ["Regression", "Binary Classification"])
state["4"]["mode"] = mode

# target encoding
if mode == "Binary Classification":
    values = df[target].dropna()

    if is_numeric_dtype(values):
        column_c0, column_i0, column_c1, column_i1 = st.columns(4)

        with column_c0:
            l_q = st.number_input(
                label="Label 0 upper limit (%)", 
                value=20, 
                min_value=0, 
                max_value=100, 
                step=1
            )
            state["4"]["l_q"] = l_q

        with column_c1:
            h_q = st.number_input(
                label="Label 1 lower limit (%)", 
                value=80, 
                min_value=0, 
                max_value=100, 
                step=1
            )
            state["4"]["h_q"] = h_q

        with column_i0:
            st.metric("Label 0 max value", f"{np.percentile(values, q=l_q):.4f}")

        with column_i1:
            st.metric("Label 1 min value", f"{np.percentile(values, q=h_q):.4f}")

    else:
        uniques = np.sort(np.unique(values)).tolist()
        col_0, col_1 = st.columns(2)

        with col_0:
            label_0 = st.selectbox("Label 0", uniques, index=0)
            state["4"]["label_0"] = label_0

        with col_1:
            label_1 = st.selectbox("Label 1", [column for column in uniques if column != label_0], index=0)
            state["4"]["label_1"] = label_1

if (
    "4" not in st.session_state["state"] or
    state["0"] != st.session_state["state"]["0"] or 
    state["1"] != st.session_state["state"]["1"] or
    state["2"] != st.session_state["state"]["2"] or
    state["3"] != st.session_state["state"]["3"] or
    state["4"] != st.session_state["state"]["4"]
):  
    # encode target if the mode is binary classification
    if state["4"]["mode"] == "Binary Classification":
        if ("l_q" in state["4"]) and ("h_q" in state["4"]):
            df = target_encode_numeric(
                df=df,
                target=target,
                l_q=state["4"]["l_q"],
                h_q=state["4"]["h_q"]
            )
        elif ("label_0" in state["4"]) and ("label_1" in state["4"]):
            df = target_encode_category(
                df=df,
                target=target,
                label_0=state["4"]["label_0"],
                label_1=state["4"]["label_1"]
            )

    # update session state
    st.session_state["state"]["mode"] = state["4"]["mode"]
    if ("l_q" in state["4"]) and ("h_q" in state["4"]):
        st.session_state["state"]["l_q"] = state["4"]['l_q']
        st.session_state["state"]["h_q"] = state["4"]['h_q']
        st.session_state["state"]["label_0"] = None
        st.session_state["state"]["label_1"] = None
    elif ("label_0" in state["4"]) and ("label_1" in state["4"]):
        st.session_state["state"]["l_q"] = None
        st.session_state["state"]["h_q"] = None
        st.session_state["state"]["label_0"] = state["4"]["label_0"]
        st.session_state["state"]["label_1"] = state["4"]["label_1"]
    else:
        st.session_state["state"]["l_q"] = None
        st.session_state["state"]["h_q"] = None
        st.session_state["state"]["label_0"] = None
        st.session_state["state"]["label_1"] = None
    st.session_state["data"]["4"] = df
    st.session_state["model"] = None
############################ STATE 4 END ##############################


# update states
for stage in ["0", "1", "2", "3", "4"]:
    st.session_state["state"][stage] = state[stage]

# data wall time
wall_time = time.time() - start_time

# read data
df = st.session_state["data"]["4"].copy().reset_index(drop=True)

# information
st.sidebar.markdown("""---""")
st.sidebar.write(f"Wall time: {wall_time:.4f} sec")
st.sidebar.write(f"Data Num: {len(df)}")
st.sidebar.write(f"Target: {st.session_state['state']['target']}")
st.sidebar.write(f"Feature Num: {len(st.session_state['state']['selected_features'])}")
st.sidebar.markdown("---")

# print df
st.write(f"Prepared data (row: {len(df)})")
st.write(df.iloc[:5])

# encoding categorical features
df = encode(df.copy(), st.session_state["state"]["encoder"])

# set final data
st.session_state["data"]["final"] = df


# train model
if st.session_state["model"] is None:
    st.markdown("""---""")
    if st.button("Start"):
        # log
        time_now = str(datetime.now())[:19]
        print(f"START | {time_now} | {get_session_id()} | {st.session_state['state']['train_file_path']}")

        # load data
        df = st.session_state["data"]["final"].copy()
        features = st.session_state["state"]["selected_features"]
        target = st.session_state["state"]["target"]
        if st.session_state["state"]["mode"] == "Regression":
            mode = "reg"
        if st.session_state["state"]["mode"] == "Binary Classification":
            mode = "clf"

        # dataset
        datasets = split_data(
            df=df,
            features=features,
            target=target,
            mode=mode,
            n_splits=st.session_state["config"]["split"]["n_splits"],
            shuffle=True,
            random_state=st.session_state["config"]["setup"]["random_state"]
        )

        # best model
        best_model, history = get_best_model(
            datasets=datasets, 
            mode=mode,
            random_state=st.session_state["config"]["setup"]["random_state"],
            n_jobs=st.session_state["config"]["setup"]["n_jobs"]
        )
        best_model["features"] = features
        best_model["target"] = target
        best_model["datasets"] = datasets

        # shap
        source, shap_value = get_shap_value(
            config=best_model,
            max_num=st.session_state["config"]["shap"]["max_num"]
        )
        output = get_importance(
            shap_value,
            sort=st.session_state["config"]["importance"]["sort"],
            normalize=st.session_state["config"]["importance"]["normalize"]
        )

        shap = {}
        shap["features"] = output["features"]
        shap["importance"] = output["importance"]
        shap["source"] = source
        shap["shap_value"] = shap_value

        if (st.session_state["state"]["auto_feature_selection"] and "random_noise" in shap["features"]):
            features = shap["features"]
            index = np.where(np.array(features)=="random_noise")[0][0]
            if index != 0:
                # print info
                st.write("Auto Feature Selection is ON.")

                # set new features
                features = features[:index]

                # dataset
                datasets = split_data(
                    df=df,
                    features=features,
                    target=target,
                    mode=mode,
                    n_splits=st.session_state["config"]["split"]["n_splits"],
                    shuffle=True,
                    random_state=st.session_state["config"]["setup"]["random_state"]
                )

                # best Model
                best_model, history = get_best_model(
                    datasets=datasets, 
                    mode=mode,
                    random_state=st.session_state["config"]["setup"]["random_state"],
                    n_jobs=st.session_state["config"]["setup"]["n_jobs"]
                )
                best_model["features"] = features
                best_model["target"] = target
                best_model["datasets"] = datasets

                # shap
                source, shap_value = get_shap_value(
                    config=best_model,
                    max_num=st.session_state["config"]["shap"]["max_num"]
                )
                output = get_importance(
                    shap_value,
                    sort=st.session_state["config"]["importance"]["sort"],
                    normalize=st.session_state["config"]["importance"]["normalize"]
                )

                shap = {}
                shap["features"] = output["features"]
                shap["importance"] = output["importance"]
                shap["source"] = source
                shap["shap_value"] = shap_value

        # update session state
        st.session_state["history"] = history
        st.session_state["model"] = best_model
        st.session_state["shap"] = shap

        # refresh page
        st_autorefresh(interval=100, limit=2)


# result
else:
    # STEP 2. Evaluation
    st.markdown('### 2. Evaluation')

    # print best model
    best = {}
    best["name"] = st.session_state["model"]["name"]
    best.update(st.session_state["model"]["score"])
    st.markdown("##### Best Model")
    st.write(best)

    # print score
    st.markdown("##### Scores")
    st.write(st.session_state["history"])

    # graph
    if st.session_state["state"]["mode"] == "Regression":
        st.altair_chart(
            plot_reg_evaluation(
                true=st.session_state["model"]["oob_true"],
                pred=st.session_state["model"]["oob_pred"],
                target=st.session_state["model"]["target"]
            ), 
            use_container_width=True
        )
    elif st.session_state["state"]["mode"] == "Binary Classification":
        st.pyplot(
            plot_confusion_matrix(
                true=st.session_state["model"]["oob_true"],
                pred=st.session_state["model"]["oob_pred"],
                target=st.session_state["model"]["target"]
            )
        )


    # STEP 3. Feature Importance
    features = st.session_state["shap"]["features"]
    importance = st.session_state["shap"]["importance"]

    col_1, col_2 = st.columns([3, 1])
    with col_1:
        st.markdown("### 3. Feature importance")
    with col_2:
        show_number = st.number_input(
            label="Number",
            value=int(np.minimum(10, len(features))),
            min_value=1,
            max_value=len(features),
            step=1
        )

    st.altair_chart(
        plot_importance(
            features=features,
            importance=importance,
            target=st.session_state["model"]["target"], 
            num=show_number
        ), 
        use_container_width=True
    )

    # download csv
    df_importance = pd.DataFrame()
    df_importance["feature"] = features
    df_importance["importance"] = importance
    st.download_button(
        label="Download (.csv)",
        data=df_importance.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"importance.csv",
        mime="text/csv"
    )


    # STEP 4. Local Explanation
    df = st.session_state["data"]["final"]
    source = st.session_state["shap"]["source"]
    shap_value = st.session_state["shap"]["shap_value"]

    col_1, col_2 = st.columns([3, 1])
    with col_1:
        st.markdown('### 4. Feature impact')
    with col_2:
        type_name = st.selectbox("Type", ["SHAP", "1D Simulation", "2D Simulation"])

    if type_name == "SHAP":
        feature = st.selectbox("Feature", features)
        st.altair_chart(
            plot_shap(
                x=source[feature].values, 
                y=shap_value[feature].values, 
                x_all=df[feature].dropna().values,
                feature=feature,
                target=st.session_state["model"]["target"],
                mean=np.mean(st.session_state["model"]["oob_true"])
            ),
            use_container_width = True
        )   

        # download csv
        df_shap = pd.DataFrame()
        df_shap[feature] = source[feature].values
        df_shap["SHAP Value"] = shap_value[feature].values
        st.download_button(
            label="Download (.csv)",
            data=df_shap.to_csv(index=False).encode("utf-8-sig"),
            file_name="shap.csv",
            mime="text/csv"
        )

        # download figures
        col_0, col_1 = st.columns(2)
        with col_0:
            if st.button("Extract all figures"):
                progress = st.progress(0)
                figs = []
                for i, feature in enumerate(features):
                    # get figure
                    figs.append(
                        matplotlib_shap(
                            x=source[feature].values, 
                            y=shap_value[feature].values, 
                            x_all=df[feature].dropna().values,
                            feature=feature,
                            target=st.session_state["model"]["target"],
                            mean=np.mean(st.session_state["model"]["oob_true"])
                        )
                    )
                    # update progress
                    progress.progress((i+1)/len(features))

                # convert to zip
                with col_1:
                    st.download_button(
                        label="Download (.zip)",
                        data=convert_figs2zip(figs),
                        file_name="shap.zip",
                        mime="application/octet-stream"
                    )

    elif type_name == "1D Simulation":
        feature = st.selectbox(
            label = "Feature", 
            options = features
        )
        x, y = simulation_1d(
            datasets=st.session_state["model"]["datasets"],
            models=st.session_state["model"]["models"],
            features=st.session_state["model"]["features"],
            feature=feature,
            mode=st.session_state["model"]["type"],
            num=st.session_state["config"]["simulation"]["num"]
        )
        st.altair_chart(
            plot_simulation_1d(
                x=x, 
                y=y,
                x_all=df[feature].dropna().values,
                feature=feature, 
                target=st.session_state["model"]["target"]
            ),
            use_container_width = True
        )

        # download csv
        df_1d = pd.DataFrame()
        df_1d[feature]=x
        df_1d["Prediction"]=y
        st.download_button(
            label="Download (.csv)",
            data=df_1d.to_csv(index=False).encode("utf-8-sig"),
            file_name="1d_simulation.csv",
            mime="text/csv"
        )

        # download figures
        col_0, col_1 = st.columns(2)
        with col_0:
            if st.button("Extract all figures"):
                progress = st.progress(0)
                figs = []
                for i, feature in enumerate(features):
                    # get x and y
                    x, y = simulation_1d(
                        datasets=st.session_state["model"]["datasets"],
                        models=st.session_state["model"]["models"],
                        features=st.session_state["model"]["features"],
                        feature=feature,
                        mode=st.session_state["model"]["type"],
                        num=st.session_state["config"]["simulation"]["num"]
                    )
                    # get figure
                    figs.append(
                        matplotlib_simulation_1d(
                            x=x, 
                            y=y,
                            x_all=df[feature].dropna().values,
                            feature=feature, 
                            target=st.session_state["model"]["target"]
                        )
                    )
                    # update progress
                    progress.progress((i+1)/len(features))

                # convert to zip
                with col_1:
                    st.download_button(
                        label = 'Download (.zip)',
                        data = convert_figs2zip(figs),
                        file_name = '1d_simulation.zip',
                        mime="application/octet-stream"
                    )


    elif type_name == "2D Simulation":
        col_0, col_1 = st.columns(2)
        with col_0:
            feature_0 = st.selectbox("Feature #1", features)
        with col_1:
            feature_1 = st.selectbox("Feature #2", [feature for feature in features if feature != feature_0])

        x_0, x_1, y = simulation_2d(
            datasets=st.session_state["model"]["datasets"],
            models=st.session_state["model"]["models"],
            features=st.session_state["model"]["features"],
            feature_0=feature_0,
            feature_1=feature_1,
            mode=st.session_state["model"]["type"],
            num=st.session_state["config"]["simulation"]["num"]
        )

        st.altair_chart(
            plot_simulation_2d(
                x_0=x_0,
                x_1=x_1,
                y=y,
                feature_0=feature_0,
                feature_1=feature_1, 
                target=st.session_state["model"]["target"]
            ),
            use_container_width = True
        )

        # download csv
        df_2d = pd.DataFrame()
        df_2d[feature_0] = x_0
        df_2d[feature_1] = x_1
        df_2d['Prediction'] = y
        st.download_button(
            label="Download (.csv)",
            data=df_2d.to_csv(index=False).encode("utf-8-sig"),
            file_name="2d_simulation.csv",
            mime="text/csv"
        )


    # STEP 5. Prediction
    st.markdown('### 5. Prediction')
    test_file_path = st.selectbox("Test Data", st.session_state["files"])

    col_0, col_1 = st.columns(2)
    with col_0:
        if st.button("Make result file"):
            # Load Data
            df_test = read_csv(
                path=test_file_path,
                max_len=None,
                add_random_noise=st.session_state["config"]["data"]["add_random_noise"],
                random_state=st.session_state["config"]["setup"]["random_state"],
            )

            # Check features
            features = st.session_state["model"]["features"]
            n_features = []       
            for feature in features:
                if feature not in list(df_test.columns):
                    n_features.append(feature)

            if len(n_features) != 0:
                st.write(f"TEST File: {test_file_path}")
                st.write(f"{n_features} are not in the test file.")
            else:
                with st.spinner(text="In progress..."):
                    # apply filter
                    df_test = apply_filter(df_test, st.session_state["state"]["filter"])

                    # nan process
                    df_data = replace_nan(df_test.copy(), st.session_state["config"]["setup"]["random_state"])

                    # encode Data
                    df_data = encode(df_data, st.session_state["state"]["encoder"])

                    # prediction
                    inputs = df_data[features].values
                    pred = []
                    for model in st.session_state["model"]["models"]:
                        if st.session_state["model"]["type"] == "reg":
                            p = model.predict(inputs)
                        elif st.session_state["model"]["type"] == "clf":
                            p = model.predict_proba(inputs)[:, 1]
                        pred.append(p)
                    pred = np.mean(pred, axis=0)

                    # Output Data
                    target = st.session_state["model"]["target"]
                    df_test[f"pred_{target}"] = pred
                    if "random_noise" in df_test.columns:
                        df_test = df_test.drop(columns=["random_noise"])

                # Download CSV
                path = test_file_path
                name = os.path.basename(path) 
           
                with col_1:
                    st.download_button(
                        label="Download (.csv)",
                        data=df_test.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"pred_{name}",
                        mime="text/csv"
                    )