import numpy as np
import pandas as pd
from pandas import DataFrame


def delete_nan(
    df: DataFrame
) -> DataFrame:
    '''Delete NaN Data'''
    return df.dropna()


def apply_replace(x, values, p):
    if pd.isna(x):
        return np.random.choice(values, 1, p=p).item()
    else:
        return x

    
def replace_nan(
    df: DataFrame,
    random_state: int=42
) -> DataFrame:
    '''Replace NaN Data using its distribution'''
    np.random.seed(random_state)
    for column in df.columns:        
        uniques, counts = np.unique(
            df[column].dropna().values, 
            return_counts = True
        )
        p = counts / np.sum(counts)
        df[column] = df[column].apply(lambda x: apply_replace(x, uniques, p))
    return df
