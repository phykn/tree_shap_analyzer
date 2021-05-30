import numpy as np

def simulator_1d(df, models, features, feature, x_min, x_max, mode='Regression', n=1000):
    index = np.where(np.array(features)==feature)[0][0]
    median = df[features].median(axis=0).values
    
    x = np.linspace(x_min, x_max, num=n)
    data = np.full([n, len(median)], median)
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

def simulator_2d(df, models, features, feature_1, x1_min, x1_max, feature_2, x2_min, x2_max, mode='Regression', n=100):
    median = df[features].median(axis=0).values
    data = np.full([n**2, len(median)], median)
    
    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, n), np.linspace(x2_min, x2_max, n))  
    index1 = np.where(np.array(features)==feature_1)[0][0]
    index2 = np.where(np.array(features)==feature_2)[0][0]
    
    data[:, index1] = x1.flatten()
    data[:, index2] = x2.flatten()
    
    y = []
    for model in models:
        if mode == 'Regression':
            pred = model.predict(data)
        elif mode == 'Classification':
            pred = model.predict_proba(data)[:, 1]
        y.append(pred)
    y = np.mean(y, axis=0)    
    return x1, x2, y.reshape(n, n)
