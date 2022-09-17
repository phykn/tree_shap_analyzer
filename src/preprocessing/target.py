import numpy as np
from pandas import DataFrame
from typing import Optional


def apply_target(
    df: DataFrame,
    target: str
) -> None:
    """delete non-valid target data"""
    return df.loc[df[target].dropna().index]


def apply_numeric(
    x: float, 
    l_v: float, 
    h_v: float
) -> Optional[int]:
    if x <= l_v:
        return 0
    elif x >= h_v:
        return 1
    else:
        return None

    
def target_encode_numeric(
    df: DataFrame,
    target: str,
    l_q: int=20,
    h_q: int=80
) -> DataFrame:
    values = df[target].dropna().values
    l_v = np.percentile(values, q=l_q)
    h_v = np.percentile(values, q=h_q)
    df[target] = df[target].apply(lambda x: apply_numeric(x, l_v, h_v))

    df = apply_target(df, target)    
    return df


def apply_category(
    x: float, 
    label_0: str, 
    label_1: str
) -> Optional[int]:
    if x == label_0:
        return 0
    elif x == label_1:
        return 1
    else:
        return None

    
def target_encode_category(
    df: DataFrame,
    target: str,
    label_0: str,
    label_1: str
) -> DataFrame:
    df[target] = df[target].apply(lambda x: apply_category(x, label_0, label_1))
    df = apply_target(df, target)    
    return df