import numpy as np
import streamlit as st
from datetime import datetime
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

def r2_score(true, pred):
    return np.corrcoef(true, pred)[0][1]**2

def mae_score(true, pred):
    return np.mean(np.abs(true - pred))

def mse_score(true, pred):
    return np.mean(np.square(true - pred))

def rmse_score(true, pred):
    return np.sqrt(np.mean(np.square(true - pred)))

def trainer(df_train, features, target, model_name='lgb', random_state=42):
    model_list = ['lgb', 'xgb', 'rf', 'et', 'gbr']
    assert model_name in model_list, f'model_name not in {model_list}.'

    x_data = df_train[features].values
    y_data = df_train[target].values.flatten()

    if model_name == 'lgb':
        model = LGBMRegressor(random_state=random_state)
    if model_name == 'xgb':
        model = XGBRegressor(random_state=random_state)
    if model_name == 'rf':
        model = RandomForestRegressor(random_state=random_state)
    if model_name == 'et':
        model = ExtraTreesRegressor(random_state=random_state)
    if model_name == 'gbr':
        model = GradientBoostingRegressor(random_state=random_state)

    model.fit(x_data, y_data)
    return model

def select_model(model_list, datas, features, target, metric='mae'):
    metrics = ['r2', 'mae', 'mse', 'rmse']
    assert metric in metrics, f'metric not in {metrics}.'

    number = len(datas)
    models, scores, oobs = [], [], []
    for model_name in model_list:
        model, oob_true, oob_pred = [], [], []
        score, i = 0, 1
        for df_train, df_valid in datas:
            # Train Model
            m = trainer(df_train, features, target, model_name=model_name)
            model.append(m)

            # Get Score
            x_valid = df_valid[features].values
            y_valid = df_valid[target].values.flatten()

            true = y_valid
            pred = m.predict(x_valid)

            oob_true += list(true)
            oob_pred += list(pred)

            if metric == 'r2':
                s = -1 * r2_score(true, pred)
            if metric == 'mae':
                s = mae_score(true, pred)
            if metric == 'mse':
                s = mse_score(true, pred)
            if metric == 'rmse':
                s = rmse_score(true, pred)

            # Log
            st.text(f'[{datetime.now()}] [{model_name}] ({i}/{number}) {metric}: {s}')

            score += s / number
            i += 1

        models.append(model)
        scores.append(score)
        oobs.append([oob_true, oob_pred])

        # Log
        st.text(f'[{datetime.now()}] [{model_name}] CV Score: {score}')
        st.text('')

    index = np.argmin(scores)
    model = models[index]
    model_name = model_list[index]
    oob = oobs[index]

    output = {}
    output['model'] = model_name
    output['weights'] =  model
    output['oob_true'] = oob[0]
    output['oob_pred'] = oob[1]
    return output