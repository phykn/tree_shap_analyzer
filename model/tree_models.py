from numpy import ndarray
from typing import Dict, Any, Optional
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier


def lgb_reg(
    x: ndarray, 
    y: ndarray,
    params: Dict[str, Any]={},
    random_state: int=42,
    n_jobs: int=-1,
) -> Dict[str, Any]:
    '''LightGBM Regressor'''

    # Set random_state and n_jobs
    params = params.copy()
    params['random_state'] = random_state
    params['n_jobs'] = n_jobs

    # model
    model = LGBMRegressor(**params)
    model.fit(x, y)
    return {
        'model': model, 
        'params': params, 
        'name': 'LGBMRegressor', 
        'type': 'reg'
    }


def lgb_clf(
    x: ndarray, 
    y: ndarray,
    params: Dict[str, Any]={},
    random_state: int=42,
    n_jobs: int=-1,
) -> Dict[str, Any]:
    '''LightGBM Classifier'''

    # Set random_state and n_jobs
    params = params.copy()
    params['random_state'] = random_state
    params['n_jobs'] = n_jobs

    # model
    model = LGBMClassifier(**params)
    model.fit(x, y)
    return {
        'model': model, 
        'params': params, 
        'name': 'LGBMClassifier', 
        'type': 'clf'
    }


def xgb_reg(
    x: ndarray, 
    y: ndarray,
    params: Dict[str, Any]={},
    random_state: int=42,
    n_jobs: Optional[int]=None,
) -> Dict[str, Any]:
    '''XGBoost Regressor'''

    # Set random_state and n_jobs
    params = params.copy()
    params['random_state'] = random_state
    params['n_jobs'] = n_jobs
    params['verbosity'] = 0

    # model
    model = XGBRegressor(**params)
    model.fit(x, y)
    return {
        'model': model, 
        'params': params, 
        'name': 'XGBRegressor', 
        'type': 'reg'
    }


def xgb_clf(
    x: ndarray, 
    y: ndarray,
    params: Dict[str, Any]={},
    random_state: int=42,
    n_jobs: Optional[int]=None,
) -> Dict[str, Any]:
    '''XGBoost Classifier'''

    # Set random_state and n_jobs
    params = params.copy()
    params['random_state'] = random_state
    params['n_jobs'] = n_jobs
    params['verbosity'] = 0
    params['use_label_encoder'] = False

    # model
    model = XGBClassifier(**params)
    model.fit(x, y)
    return {
        'model': model, 
        'params': params, 
        'name': 'XGBClassifier', 
        'type': 'clf'
    }


def rf_reg(
    x: ndarray, 
    y: ndarray,
    params: Dict[str, Any]={},
    random_state: int=42,
    n_jobs: int=-1,
) -> Dict[str, Any]:
    '''Random Forest Regressor'''

    # Set random_state and n_jobs
    params = params.copy()
    params['random_state'] = random_state
    params['n_jobs'] = n_jobs

    # model
    model = RandomForestRegressor(**params)
    model.fit(x, y)
    return {
        'model': model, 
        'params': params, 
        'name': 'RandomForestRegressor', 
        'type': 'reg'
    }


def rf_clf(
    x: ndarray, 
    y: ndarray,
    params: Dict[str, Any]={},
    random_state: int=42,
    n_jobs: int=-1,
) -> Dict[str, Any]:
    '''Random Forest Classifier'''

    # Set random_state and n_jobs
    params = params.copy()
    params['random_state'] = random_state
    params['n_jobs'] = n_jobs

    # model
    model = RandomForestClassifier(**params)
    model.fit(x, y)
    return {
        'model': model, 
        'params': params, 
        'name': 'RandomForestClassifier', 
        'type': 'clf'
    }


def et_reg(
    x: ndarray, 
    y: ndarray,
    params: Dict[str, Any]={},
    random_state: int=42,
    n_jobs: int=-1,
) -> Dict[str, Any]:
    '''Extra Trees Regressor'''

    # Set random_state and n_jobs
    params = params.copy()
    params['random_state'] = random_state
    params['n_jobs'] = n_jobs

    # model
    model = ExtraTreesRegressor(**params)
    model.fit(x, y)
    return {
        'model': model, 
        'params': params, 
        'name': 'ExtraTreesRegressor', 
        'type': 'reg'
    }


def et_clf(
    x: ndarray, 
    y: ndarray,
    params: Dict[str, Any]={},
    random_state: int=42,
    n_jobs: int=-1,
) -> Dict[str, Any]:
    '''Extra Trees Classifier'''

    # Set random_state and n_jobs
    params = params.copy()
    params['random_state'] = random_state
    params['n_jobs'] = n_jobs

    # model
    model = ExtraTreesClassifier(**params)
    model.fit(x, y)
    return {
        'model': model, 
        'params': params, 
        'name': 'ExtraTreesClassifier', 
        'type': 'clf'
    }
