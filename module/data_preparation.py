from sklearn.model_selection import KFold

def select_feature(df, features, target):
    assert len(target)==1, 'The length of a target list is not one.'
    out = df.loc[~df[target[0]].isnull()].reset_index(drop=True)
    return out[features+target]

def split_data(df, n_splits=5, shuffle=True, random_state=42):
    data = []
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_index, valid_index in kf.split(df):
        # Split Data
        df_train = df.iloc[train_index]
        df_valid = df.iloc[valid_index]

        # Fill Na
        mean = df_train.mean()
        df_train = df_train.fillna(mean)
        df_valid = df_valid.fillna(mean)

        data.append([df_train, df_valid])
    return data