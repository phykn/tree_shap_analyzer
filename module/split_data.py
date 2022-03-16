import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

def split_data(df, features, targets, mode, n_splits=5, shuffle=True, random_state=42):
    df = df.copy()
    df = df[features+targets]

    if mode == 'Regression':
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split = kf.split(df)

    elif mode == 'Classification':
        x = df[features].values
        y = df[targets].values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split = skf.split(x, y)

    data_sets = []
    for train_index, valid_index in split:
        # Split Data
        df_train = df.iloc[train_index]
        df_valid = df.iloc[valid_index]
        data_sets.append([df_train, df_valid])
    return data_sets