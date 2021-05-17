import numpy as np
import shap
shap.initjs()

def get_shap_value(weights, datas, features):
    shap_value = []
    df_valids = [df_valid for df_train, df_valid in datas]
    for weight, df_valid in zip(weights, df_valids):
        e = shap.TreeExplainer(weight)
        s = e.shap_values(df_valid[features].values)
        shap_value.append(s)
    shap_value = np.concatenate(shap_value, axis=0)
    return shap_value

def get_feature_importance(shap_value, features, sort=False):
    feature_names = np.array(features)
    feature_importances = np.sum(np.abs(shap_value), axis=0)
    feature_importances = 100 * feature_importances / np.sum(feature_importances)
    if sort:
        index = np.argsort(feature_importances)
        feature_names = feature_names[index]
        feature_importances = feature_importances[index]
    return feature_names[::-1], feature_importances[::-1]