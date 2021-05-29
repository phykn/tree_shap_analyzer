import numpy as np

def simulator(df, models, features, feature, x_min, x_max, mode='Regression', n=1000):
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