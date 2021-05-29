from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier

def trainer(df, features, targets, model_name='lgb_reg', random_state=42, n_jobs=-1):
    model_list = [
        'lgb_reg', 'xgb_reg', 'rf_reg', 'et_reg',
        'lgb_clf', 'xgb_clf', 'rf_clf', 'et_clf',
    ]
    assert model_name in model_list, f'model_name not in {model_list}.'

    x_data = df[features].values
    y_data = df[targets].values.flatten()

    # Regression
    if model_name == 'lgb_reg':
        model = LGBMRegressor(random_state=random_state, n_jobs=n_jobs)
    if model_name == 'xgb_reg':
        model = XGBRegressor(random_state=random_state, verbosity=0, n_jobs=n_jobs)
    if model_name == 'rf_reg':
        model = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
    if model_name == 'et_reg':
        model = ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs)

    # Classification
    if model_name == 'lgb_clf':
        model = LGBMClassifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == 'xgb_clf':
        model = XGBClassifier(use_label_encoder=False, verbosity=0, random_state=random_state, n_jobs=n_jobs)
    if model_name == 'rf_clf':
        model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == 'et_clf':
        model = ExtraTreesClassifier(random_state=random_state, n_jobs=n_jobs)

    model.fit(x_data, y_data)
    return model