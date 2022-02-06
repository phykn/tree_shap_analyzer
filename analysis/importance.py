import numpy as np
from typing import Dict, List
from pandas import DataFrame

def get_importance(
    shap_value: DataFrame, 
    sort: bool=False, 
    normalize: bool=False
) -> Dict[str, List]:
    '''Get feature importance from shap value'''
    features = np.array(shap_value.columns)
    importance = np.sum(np.abs(shap_value.values), axis=0)

    # Sort importance
    if sort:
        index = np.argsort(importance)
        features = features[index]
        importance = importance[index]   

    # Normalize
    if normalize:
        importance = 100 * importance / np.sum(importance)    

    features = list(features[::-1])
    importance = list(importance[::-1])

    return {
        'features': features, 
        'importance': importance
    }