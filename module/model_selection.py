import numpy as np
import streamlit as st
from datetime import datetime
from .metrics import *
from .trainer import trainer

def select_model(model_names, datas, features, targets, metric='MAE', n_jobs=-1):
    metrics = ['R2', 'MAE', 'MSE', 'RMSE', 'AUC', 'LOGLOSS', 'ACCURACY']
    assert metric in metrics, f'metric not in {metrics}.'
    
    number = len(datas)
    models, scores, oobs = [], [], []
    for model_name in model_names:
        model, oob_true, oob_pred = [], [], []
        score, i = 0, 1
        for df_train, df_valid in datas:
            # Train Model
            m = trainer(df_train, features, targets, model_name=model_name, n_jobs=n_jobs)
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

            # Log
            st.info(f'{datetime.now()} | {model_name} | {i}/{number} | {metric} = {np.abs(s)}')

            score += s / number
            i += 1

        models.append(model)
        scores.append(score)
        oobs.append([oob_true, oob_pred])

        # Log
        st.success(f'{datetime.now()} | {model_name} | CV Score = {np.abs(score)}')
        st.text('')

    index = np.argmin(scores)
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
    return output
