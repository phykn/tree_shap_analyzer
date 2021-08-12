import numpy as np
import streamlit as st
from stqdm import stqdm
from datetime import datetime
from .metrics import *
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier

def trainer(df, features, targets, model_name='lgb_reg', random_state=42, n_jobs=-1):
    model_list = ['lgb_reg', 'xgb_reg', 'rf_reg', 'et_reg',
                  'lgb_clf', 'xgb_clf', 'rf_clf', 'et_clf']
    assert model_name in model_list, f'model_name not in {model_list}.'

    # Regression
    if model_name == 'lgb_reg':
        model = LGBMRegressor(random_state=random_state, n_jobs=n_jobs)
    if model_name == 'xgb_reg':
        model = XGBRegressor(random_state=random_state, verbosity=0, n_jobs=n_jobs)
    if model_name == 'rf_reg':
        model = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
    if model_name == 'et_reg':
        model = ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs)

    # Classification
    if model_name == 'lgb_clf':
        model = LGBMClassifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == 'xgb_clf':
        model = XGBClassifier(use_label_encoder=False, verbosity=0, random_state=random_state, n_jobs=n_jobs)
    if model_name == 'rf_clf':
        model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == 'et_clf':
        model = ExtraTreesClassifier(random_state=random_state, n_jobs=n_jobs)

    model.fit(df[features], df[targets])
    return model

def select_model(model_names, datas, features, targets, metric='MAE', random_state=42, n_jobs=-1):
    metrics = ['R2', 'MAE', 'MSE', 'RMSE', 'AUC', 'LOGLOSS', 'ACCURACY']
    assert metric in metrics, f'metric not in {metrics}.'
    
    number = len(datas)
    models, scores, oobs = [], [], []
    for model_name in stqdm(model_names, desc='Progress'):
        model, oob_true, oob_pred = [], [], []
        score = 0
        for df_train, df_valid in datas:
            # Train Model
            m = trainer(df_train, 
                        features, 
                        targets, 
                        model_name   = model_name, 
                        random_state = random_state,
                        n_jobs       = n_jobs)
            model.append(m)

            # Get Score
            x_valid = df_valid[features].values
            y_valid = df_valid[targets].values.flatten()

            true = y_valid
            if '_reg' in model_name:
                mode = 'Regression'
                pred = m.predict(x_valid)
            elif '_clf' in model_name:
                mode = 'Classification'       
                pred = m.predict_proba(x_valid)[:, 1] 

            oob_true += list(true)
            oob_pred += list(pred)

            if metric == 'R2':
                s = -1 * r2_score(true, pred)
            if metric == 'MAE':
                s = mae_score(true, pred)
            if metric == 'MSE':
                s = mse_score(true, pred)
            if metric == 'RMSE':
                s = rmse_score(true, pred)
            if metric == 'AUC':
                s = -1 * auc_score(true, pred)
            if metric == 'LOGLOSS':
                s = logloss_score(true, pred)
            if metric == 'ACCURACY':
                s = -1 * accuracy_score(true, pred)

            score += s / number

        models.append(model)
        scores.append(score)
        oobs.append([oob_true, oob_pred])

        # Log
        st.success(f'{datetime.now()} | {model_name} | CV Score = {np.abs(score)}')

    index = np.argmin(scores)
    score = scores[index]
    model = models[index]
    model_name = model_names[index]
    oob = oobs[index]

    output = {}
    output['mode'] = mode
    output['fold'] = number
    output['model_name'] = model_name
    output['features'] = features
    output['targets'] = targets
    output['models'] =  model
    output['oob_true'] = np.array(oob[0])
    output['oob_pred'] = np.array(oob[1])
    output['cv_score'] = score
    return output