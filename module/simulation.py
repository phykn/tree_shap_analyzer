import numpy as np

def simulator(df, output, features, feature, x_min, x_max, n=1000):
    index = np.where(np.array(features)==feature)[0][0]
    means = df[features].mean(axis=0).values
    
    x = np.linspace(x_min, x_max, num=n)
    data = np.full([n, len(means)], means)
    data[:, index] = x
    
    y = []
    for weight in output['weights']:
        y.append(weight.predict(data))
    y = np.mean(y, axis=0)
    return x, y