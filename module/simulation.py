import numpy as np

def simulator_1d(df, models, features, feature, mode='Regression', n=1000):
    x = df[feature].dropna().values    
    x = np.linspace(np.min(x), np.max(x), num=n)

    replace = df[features].mean(axis=0).values
    index   = np.where(np.array(features)==feature)[0][0]
    data    = np.full([n, len(replace)], replace)
    data[:, index] = x
    
    y = []
    for model in models:
        if mode == 'Regression':
            pred = model.predict(data)
        elif mode == 'Classification':
            pred = model.predict_proba(data)[:, 1]
        y.append(pred)
    y = np.mean(y, axis=0)
    return x, y

def simulator_2d(df, models, features, feature_1, feature_2, mode='Regression', n=100):
    x1 = df[feature_1].dropna().values
    x2 = df[feature_2].dropna().values
    x1, x2 = np.meshgrid(np.linspace(np.min(x1), np.max(x1), num=n), 
                         np.linspace(np.min(x2), np.max(x2), num=n))
    x1 = x1.flatten()
    x2 = x2.flatten()

    replace = df[features].mean(axis=0).values  
    index1  = np.where(np.array(features)==feature_1)[0][0]
    index2  = np.where(np.array(features)==feature_2)[0][0]
    data    = np.full([n**2, len(replace)], replace)       
    data[:, index1] = x1
    data[:, index2] = x2
    
    y = []
    for model in models:
        if mode == 'Regression':
            pred = model.predict(data)
        elif mode == 'Classification':
            pred = model.predict_proba(data)[:, 1]
        y.append(pred)
    y = np.mean(y, axis=0)    
    return x1, x2, y