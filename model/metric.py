import numpy as np
from sklearn.metrics import roc_auc_score

def r2_score(
    true: np.ndarray, 
    pred: np.ndarray
) -> float:
    return np.corrcoef(true, pred)[0][1]**2

def mae_score(
    true: np.ndarray, 
    pred: np.ndarray
) -> float:
    return np.mean(np.abs(true - pred))

def mse_score(
    true: np.ndarray, 
    pred: np.ndarray
) -> float:
    return np.mean(np.square(true - pred))

def rmse_score(
    true: np.ndarray, 
    pred: np.ndarray
) -> float:
    return np.sqrt(np.mean(np.square(true - pred)))

def mape_score(
    true: np.ndarray, 
    pred: np.ndarray,
    epsilon: float=1e-7
) -> float:
	return 100 * np.mean(np.abs((true-pred)/(true+epsilon)))

def auc_score(
    true: np.ndarray, 
    pred: np.ndarray
) -> float:
    return roc_auc_score(true, pred)

def logloss_score(
    true: np.ndarray, 
    pred: np.ndarray,
    epsilon: float=1e-7
) -> float:
    return -1*np.mean(true*np.log(pred+epsilon)+(1-true)*np.log(1-pred+epsilon))

def accuracy_score(
    true: np.ndarray, 
    pred: np.ndarray
) -> float:
    true = np.round(true)
    pred = np.round(pred)
    return np.mean(true==pred)