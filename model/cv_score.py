import numpy as np
from typing import Dict, Any, Optional
from .metric import *

def cv_score(
    model_func, 
    datasets, 
    params={},
    random_state: int=42,
    n_jobs: Optional[int]=None,
) -> Dict[str, Any]:
    '''
    model: Tree Model Function
    datasets: List of dataset.
    dataset = {
        'x_train': ndarray,
        'y_train': ndarray,
        'x_valid': ndarray,
        'y_valid': ndarray,
    }
    '''
    models = []
    oob_true = []
    oob_pred = []
    for dataset in datasets:
        x_train = dataset['x_train']
        y_train = dataset['y_train'].flatten()
        x_valid = dataset['x_valid']
        y_valid = dataset['y_valid'].flatten()

        output = model_func(
            x = x_train, 
            y = y_train, 
            params = params,
            random_state = random_state,
            n_jobs = n_jobs
        )
        model = output['model']
        if output['type'] == 'reg':
            pred = model.predict(x_valid)
        elif output['type'] == 'clf':
            pred = model.predict_proba(x_valid)[:, 1]
        else:
            raise ValueError

        models.append(model)
        oob_true.append(y_valid)
        oob_pred.append(pred)

    oob_true = np.concatenate(oob_true, axis=0)
    oob_pred = np.concatenate(oob_pred, axis=0)

    score = {}
    if output['type'] == 'reg':
        score['r2'] = r2_score(oob_true, oob_pred)
        score['mae'] = mae_score(oob_true, oob_pred)
        score['mse'] = mse_score(oob_true, oob_pred)
        score['rmse'] = rmse_score(oob_true, oob_pred)
        score['mape'] = mape_score(oob_true, oob_pred)
    elif output['type'] == 'clf':
        score['auc'] = auc_score(oob_true, oob_pred)
        score['logloss'] = logloss_score(oob_true, oob_pred)
        score['accuracy'] = accuracy_score(oob_true, oob_pred)

    else:
        raise ValueError 
    
    return {
        'name': output['name'],
        'type': output['type'],
        'models': models,
        'oob_true': oob_true,
        'oob_pred': oob_pred,
        'score': score
    }