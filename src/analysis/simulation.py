import numpy as np
import pandas as pd
from numpy import ndarray
from typing import Tuple, List, Dict, Any


def simulation_1d(
    datasets: List[Dict[str, ndarray]],
    models: List[Any],
    features: List[str], 
    feature: str, 
    mode: str="reg", 
    num: int=1000
) -> Tuple[ndarray, ndarray]:
    # dataframe
    df = pd.DataFrame(
        data = np.concatenate(
            [dataset["x_valid"] for dataset in datasets],
            axis=0
        ),
        columns=features
    )

    # feature space
    x = df[feature].dropna().values    
    x = np.linspace(np.min(x), np.max(x), num=num)

    replace = df[features].mean(axis=0).values
    index = np.where(np.array(features)==feature)[0][0]
    data = np.full([num, len(replace)], replace)
    data[:, index] = x
    
    # predict
    y = []
    for model in models:
        if mode == "reg":
            pred = model.predict(data)
        elif mode == "clf":
            pred = model.predict_proba(data)[:, 1]
        y.append(pred)
    y = np.mean(y, axis=0)

    return x, y


def simulation_2d(
    datasets: List[Dict[str, ndarray]],
    models: List[Any],
    features: List[str], 
    feature_0: str, 
    feature_1: str, 
    mode: str="reg", 
    num: int=1000
) -> Tuple[ndarray, ndarray, ndarray]:    
    # dataframe
    df = pd.DataFrame(
        data = np.concatenate(
            [dataset["x_valid"] for dataset in datasets],
            axis=0
        ),
        columns=features
    )    

    x_0 = df[feature_0].dropna().values
    x_1 = df[feature_1].dropna().values
    x_0, x_1 = np.meshgrid(
        np.linspace(np.min(x_0), np.max(x_0), num=num), 
        np.linspace(np.min(x_1), np.max(x_1), num=num)
    )
    x_0 = x_0.flatten()
    x_1 = x_1.flatten()

    replace = df[features].mean(axis=0).values  
    index_0 = np.where(np.array(features)==feature_0)[0][0]
    index_1 = np.where(np.array(features)==feature_1)[0][0]
    data = np.full([num**2, len(replace)], replace)       
    data[:, index_0] = x_0
    data[:, index_1] = x_1
    
    # predict
    y = []
    for model in models:
        if mode == "reg":
            pred = model.predict(data)
        elif mode == "clf":
            pred = model.predict_proba(data)[:, 1]
        y.append(pred)
    y = np.mean(y, axis=0)  

    return x_0, x_1, y