import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

def del_outlier(df, lower_limit=0.01, upper_limit=0.99):
    df_out = df.copy()
    l_lims = df.quantile(q=lower_limit)
    u_lims = df.quantile(q=upper_limit)
    for column in df.columns:
        values = df[column].values.astype(float)
        values[values < l_lims[column]] = np.nan
        values[values > u_lims[column]] = np.nan
        df_out[column] = values
    return df_out

def select_feature(df, features, targets):
    assert len(targets)==1, 'The length of a target list is not one.'
    out = df.loc[~df[targets[0]].isnull()].reset_index(drop=True)
    return out[features+targets]

def split_data(df, features, targets, mode, n_splits=5, shuffle=True, random_state=42):
    if mode == 'Regression':
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split = kf.split(df)

    elif mode == 'Classification':
        x = df[features].values
        y = df[targets].values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split = skf.split(x, y)

    data = []
    for train_index, valid_index in split:
        # Split Data
        df_train = df.iloc[train_index]
        df_valid = df.iloc[valid_index]

        # Fill Na
        median = df_train.median()
        df_train = df_train.fillna(median)
        df_valid = df_valid.fillna(median) 

        data.append([df_train, df_valid])
    return data