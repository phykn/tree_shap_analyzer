# Tree SHAP analyzer
<figure>
    <img src="image/front.png" width="80%">  
</figure>  

Tree SHAP analyzer is web app data analyzer. Our goal is to democratize data analysis for those who are not familiar with writing code. It includes data preprocessing, model selection, model optimization, and prediction. Please check the details for the SHAP (https://github.com/slundberg/shap).

# How to start
To start is app, run the `main.py` according to the script below.  

```python
streamlit run main.py
```

# How to analyze
This app has five sections which are `data preparation`, `evaluation`, `feature importance`, `feature dependence` and `prediction`.  

## Data preparation
- **Train data file**: Select the data file to use for the analysis.
<figure>
    <img src="image/train_data.png" width="70%">  
</figure>  

- **Data preprocessing**: This app provides some simple data preprocessing.  
    - **Missing Value**: Select treatment method for missing values in the train data.
      - **Delete**: Delete the row containing the missing values.
      - **Replace**: Replace with any value in the same column.
      <figure>
          <img src="image/missing_value.png" width="30%">  
      </figure>  

    - **Filter**: Filter is the process of choosing a specific subset of the train data.
      - **Filter Number**: Select number of columns to apply filter.
      <figure>
          <img src="image/filter_num.png" width="30%">  
      </figure>  

      - **Select values**: Select values for the filtered column.
        - **Numeric data**: You can select data range with the slider.
        <figure>
            <img src="image/filter_range.png" width="70%">  
        </figure> 

        - **Categorical data**: You can select specific data with the multiselect.
        <figure>
            <img src="image/filter_categorical.png" width="70%">  
        </figure>  

    - **Feature selection**: Select features using the check boxes to apply to the analysis. Data quality shows usable data ratio (except NaN data). `1.0` means all data is avaliable. Please check about `random_noise` in the `Auto Feature Selection` section.
      <figure>
          <img src="image/feature.png" width="30%">  
      </figure>  

    - **Target / Solver**: Select a target column and a solver.
      - **Regression**: `Regression` solver solves regression problems. It uses models as `LGBMRegressor`, `XGBRegressor`, `RandomForestRegressor` and `ExtraTreesRegressor`.
      <figure>
          <img src="image/solver_regression.png" width="70%">  
      </figure>  

      - **Binary Classification**: `Binary Classification` solver solves classification problems. It uses models as `LGBMClassifier`, `XGBClassifier`, `RandomForestClassifier` and `ExtraTreesClassifier`.  Here, the target needs to be encoded. Set a percentage to separate the target by 0 and 1.
      <figure>
          <img src="image/solver_classification.png" width="70%">  
      </figure>  

    - **Check dataset**: You can check the prepared data set in real time.
    <figure>
        <img src="image/check_dataset.png" width="70%">
    </figure>  

## Evaluation
When you click `Start` button, the app train four models and select the best automatically. For the regression models, the minimum `mae` (mean average error) model is selected. For the classification models, the maximum `auc` (area under the curve) model is selected. It provides visualization of prediction results and answers.  

|Regression|Binary Calssification|
|:---:|:---:|
|<img src="image/result_regression.png">|<img src="image/result_classification.png">|

## Feature importance
In this section, you can see feature imporance of the best model. You can choose the number of features in the graph with the number input. When you click the `Download` button, you can download the feature importance data as a `csv` file.
<figure>
    <img src="image/feature_importance.png" width="50%">  
</figure> 

## Feature dependence
|SHAP|1D simulation|2D simulation|
|:---:|:---:|:---:|
|<img src="image/feature_dependence_shap.png" width="100%">|<img src="image/feature_dependence_1d.png" width="90%">|<img src="image/feature_dependence_2d.png" width="97%">|


## Prediction
## Option

# Reference
1. SHAP: https://github.com/slundberg/shap
2. Titanic dataset: https://www.kaggle.com/c/titanic
3. Boston house dataset: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
