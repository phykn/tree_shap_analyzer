import numpy as np
import pandas as pd
import shap
from datetime import datetime
shap.initjs()

def get_shap_value(model, df, features, approximate=False, max_num=1000):
    if len(df) > max_num:
        df = df.sample(n=max_num, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(
        df[features].values, 
        approximate=approximate, 
        check_additivity=False
    )
    return shap_value, df

def get_feature_importance(shap_value, features, sort=False):
    feature_names = np.array(features)
    feature_importances = np.sum(np.abs(shap_value), axis=0)
    feature_importances = 100 * feature_importances / np.sum(feature_importances)
    if sort:
        index = np.argsort(feature_importances)
        feature_names = feature_names[index]
        feature_importances = feature_importances[index]        
    return feature_names[::-1], feature_importances[::-1]

def get_shap_value_from_output(output, datas, max_num=1000):
    model_name = output['model_name']
    models = output['models']
    features = output['features']

    shap_source, shap_values = [], []
    dfs = [df for _, df in datas]
    for model, df in zip(models, dfs):
        if model_name in ['lgb_reg', 'lgb_clf']:
            approximate = False
        else:
            approximate = True

        shap_value, df = get_shap_value(
            model, 
            df, 
            features, 
            approximate=approximate, 
            max_num=max_num
        )

        shap_source.append(df)
        if model_name in ['lgb_clf', 'rf_clf', 'et_clf']:
            shap_values.append(shap_value[1])
        else:
            shap_values.append(shap_value)
        
    shap_source = pd.concat(shap_source, axis=0)
    shap_values = np.concatenate(shap_values, axis=0)
    return shap_source, shap_values

def get_important_feature(feature_names, feature_importances, cut=100):
    np.random.seed(42)
    
    cumulative_importances = [0]
    for feature_importance in feature_importances:
        val = cumulative_importances[-1] + feature_importance
        cumulative_importances.append(val)
    cumulative_importances = np.array(cumulative_importances[1:])
    
    index = np.where(cumulative_importances <= cut)
    feature_names = feature_names[index]
    np.random.shuffle(feature_names)
    feature_names = list(feature_names)
    return feature_names