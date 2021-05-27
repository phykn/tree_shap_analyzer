import pandas as pd
import numpy as np
import shap
shap.initjs()

def get_shap_value(weights, datas, features, max_num=1000):
    shap_source, shap_value = [], []
    df_valids = [df_valid for df_train, df_valid in datas]
    for weight, df_valid in zip(weights, df_valids):
        if len(df_valid) > max_num:
            df_valid = df_valid.sample(n=max_num)
        e = shap.TreeExplainer(weight)
        s = e.shap_values(df_valid[features].values)
        shap_source.append(df_valid)
        shap_value.append(s)
    shap_source = pd.concat(shap_source, axis=0)
    shap_value = np.concatenate(shap_value, axis=0)
    return shap_source, shap_value

def get_feature_importance(shap_value, features, sort=False):
    feature_names = np.array(features)
    feature_importances = np.sum(np.abs(shap_value), axis=0)
    feature_importances = 100 * feature_importances / np.sum(feature_importances)
    if sort:
        index = np.argsort(feature_importances)
        feature_names = feature_names[index]
        feature_importances = feature_importances[index]
    return feature_names[::-1], feature_importances[::-1]