from typing import List, Dict
from pandas import DataFrame
from sklearn.model_selection import KFold, StratifiedKFold


def split_data(
    df: DataFrame, 
    features: List[str], 
    target: str, 
    mode: str, 
    n_splits: int=5, 
    shuffle: bool=True, 
    random_state: int=42
) -> List[Dict[str, DataFrame]]:
    assert n_splits > 1, 'n_splits should be grater than 1.'
    df = df[features+[target]]
    df = df.reset_index(drop=True)

    if mode == 'reg':
        kf = KFold(
            n_splits = n_splits, 
            shuffle = shuffle, 
            random_state = random_state
        )
        split = kf.split(df)

    elif mode == 'clf':
        x = df[features].values
        y = df[[target]].values
        skf = StratifiedKFold(
            n_splits = n_splits, 
            shuffle = shuffle, 
            random_state = random_state
        )
        split = skf.split(x, y)

    datasets = []
    for train_index, valid_index in split:
        df_train = df.iloc[train_index]
        df_valid = df.iloc[valid_index]
        dataset = {
            'x_train': df_train[features].values,
            'y_train': df_train[[target]].values, 
            'x_valid': df_valid[features].values,
            'y_valid': df_valid[[target]].values
        }
        datasets.append(dataset)
    return datasets
