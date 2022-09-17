import streamlit as st
import numpy as np
import pandas as pd
from numpy import ndarray
from typing import List, Dict, Any
from .tree_models import *
from .cv_score import cv_score


def get_best_model(
    datasets: List[Dict[str, ndarray]], 
    mode: str="reg",
    random_state: int=42,
    n_jobs: Optional[int]=None,
) -> Dict[str, Any]:
    # init progress bar
    progress = 0.0
    bar = st.progress(progress)

    # get best model
    if mode == "reg":
        best_model = None
        history = []    
        score = np.inf        
        for model_func in [lgb_reg, xgb_reg, rf_reg, et_reg]:
            output = cv_score(
                model_func=model_func, 
                datasets=datasets,
                random_state=random_state,
                n_jobs=n_jobs
            )

            # write history
            history.append(pd.DataFrame({output["name"]: output["score"]}).T)

            # update best_model
            if output["score"]["mae"] < score:
                score = output["score"]["mae"]
                best_model = output

            # update progress bar
            progress += 0.25
            bar.progress(progress)

        history = pd.concat(history, axis=0)
        history = history.sort_values(by=["mae"], ascending=True, axis=0)

    elif mode == "clf":
        best_model = None
        history = []    
        score = 0
        for model_func in [lgb_clf, xgb_clf, rf_clf, et_clf]:
            output = cv_score(model_func, datasets)

            # update best_model
            if output["score"]["auc"] > score:
                score = output["score"]["auc"]
                best_model = output

            # write history
            history.append(pd.DataFrame({output["name"]: output["score"]}).T)

            # update progress bar
            progress += 0.25
            bar.progress(progress)

        history = pd.concat(history, axis=0)
        history = history.sort_values(by=["auc"], ascending=False, axis=0)

    else:
        raise ValueError

    return best_model, history