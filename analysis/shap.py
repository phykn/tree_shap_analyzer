import numpy as np
import pandas as pd
import shap
from typing import Tuple
from numpy import ndarray
from pandas import DataFrame

def _shap_value(
    model, 
    x_data: ndarray, 
    approximate: bool=False, 
    max_num: int=1000
) -> np.ndarray:
    '''Get shap value of the model'''
    if len(x_data) > max_num:
        x_data = x_data[:max_num]

    explainer = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(
        X = x_data,
        approximate = approximate,
        check_additivity = False
    )
    return x_data, shap_value


def get_shap_value(
    config,
    max_num: int=1000
) -> Tuple[DataFrame]:
    # Load Data
    name = config['name']
    models = config['models']
    features = config['features']
    datasets = config['datasets']

    # Shap Value
    source = []
    shap_value = []
    x_datas = [dataset['x_valid'] for dataset in datasets]    
    for model, x_data in zip(models, x_datas):
        if name in [
            'LGBMRegressor', 
            'LGBMClassifier'
        ]:
            approximate = False
        elif name in [
            'XGBRegressor', 
            'XGBClassifier', 
            'RandomForestRegressor', 
            'RandomForestClassifier', 
            'ExtraTreesRegressor', 
            'ExtraTreesClassifier'
        ]:
            approximate = True
        else:
            raise ValueError

        x_data, sv = _shap_value(
            model = model,
            x_data = x_data, 
            approximate = approximate,
            max_num = max_num
        )
        if name in [
            'LGBMClassifier', 
            'RandomForestClassifier', 
            'ExtraTreesClassifier'
        ]: 
            sv = sv[1]

        source.append(x_data)
        shap_value.append(sv)

    source = pd.DataFrame(
        data = np.concatenate(source, axis=0),
        columns = features
    )
    shap_value = pd.DataFrame(
        data = np.concatenate(shap_value, axis=0),
        columns = features
    )

    return source, shap_value