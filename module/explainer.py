import numpy as np
import pandas as pd
import shap
shap.initjs()

def get_shap_value(model, df, features, approximate=False, max_num=1000, random_state=42):
    df = df.sample(n=max_num, random_state=random_state) if len(df) > max_num else df
    
    explainer  = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(df[features].values, 
                                       approximate      = approximate, 
                                       check_additivity = False)
    return shap_value, df

def get_shap_value_from_output(datas, output, max_num=1000, random_state=42):
    model_name  = output['model_name']
    models      = output['models']
    features    = output['features']
    shap_source = []
    shap_values = []
    df_valids   = [df_valid for df_train, df_valid in datas]

    for model, df_valid in zip(models, df_valids):
        if model_name in ['lgb_reg', 'lgb_clf']:
            approximate = False
        else:
            approximate = True

        shap_value, df_valid = get_shap_value(model,
                                              df_valid,
                                              features, 
                                              approximate  = approximate,
                                              max_num      = max_num,
                                              random_state = random_state)    
        shap_source.append(df_valid)      
        if model_name in ['lgb_clf', 'rf_clf', 'et_clf']:
            shap_values.append(shap_value[1])
        else:
            shap_values.append(shap_value)
                          
    shap_source = pd.concat(shap_source, axis=0)
    shap_values = pd.DataFrame(data    = np.concatenate(shap_values, axis=0), 
                               columns = features)
    return shap_source, shap_values

def get_feature_importance(shap_values, sort=False, normalize=False):
    feature_names = np.array(shap_values.columns)
    feature_importances = np.sum(np.abs(shap_values.values), axis=0)
    if sort:
        index = np.argsort(feature_importances)
        feature_names = feature_names[index]
        feature_importances = feature_importances[index]   

    if normalize:
        feature_importances = len(feature_importances) * feature_importances / np.sum(feature_importances)    
        if 'random_noise' in feature_names:
            feature_importances /= feature_importances[np.where(feature_names == 'random_noise')]
    else:
        feature_importances = 100 * feature_importances / np.sum(feature_importances)
    return list(feature_names[::-1]), list(feature_importances[::-1])