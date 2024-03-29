# Tree SHAP analyzer
Tree SHAP analyzer is web based data analyzer. Our goal is to democratize data analysis for those who are not familiar with writing code. It includes data preprocessing, model selection, model optimization, and prediction. This allows you to see which features are important and how they affect the target.

<details>
  <summary>example</summary>
  <figure>
      <img src="image/front.png">  
  </figure>
</details>  

## Requirements
<details>
  <summary>show list</summary>  
  
  - numpy
  - pandas
  - omegaconf
  - matplotlib
  - seaborn
  - scikit-learn
  - lightgbm
  - xgboost
  - shap
  - altair
  - streamlit
  - streamlit-autorefresh
</details>  

## How to run
To start this app, run the `main.py` according to the script below.  

```python
streamlit run main.py
```

## How to use
This app has five sections which are `Data preparation`, `Evaluation`, `Feature importance`, `Feature dependence` and `Prediction`.  

### 1. Data preparation
- **Train data file**: Select the data file to use for the analysis.
  <details>
    <summary>example</summary>
    <figure>
        <img src="image/train_data.png">  
    </figure>  
  </details>  

- **Data preprocessing**: It provides several simple data preprocessing.  
    - **Missing Value**: Select treatment method for missing values in the train data.
      <details>
        <summary>example</summary> 
        <figure>
            <img src="image/missing_value.png">
        </figure>
      </details>  

      - **Delete**: It deletes the row containing the missing values.
      - **Replace**: It replaces with any value in the same column.  

    - **Filter**: Filter is the process of choosing a specific subset of the train data.
      - **Filter Number**: Select number of columns to apply filter.
        <details>
          <summary>example</summary> 
          <figure>
              <img src="image/filter_num.png">
          </figure>  

      - **Select values**: Select values for the filtered column.
        - **Numeric data**: You can select data range with the slider.
          <details>
            <summary>example</summary> 
            <figure>
                <img src="image/filter_range.png">  
            </figure> 
          </details>

        - **Categorical data**: You can select specific data with the multiselect.
          <details>
            <summary>example</summary> 
            <figure>
                <img src="image/filter_categorical.png">  
            </figure> 
          </details>  

    - **Feature selection**: Select features using the check boxes to apply to the analysis. Data quality shows usable data ratio (except NaN data). `1.0` means all data is avaliable. Please check about `random_noise` in the `Auto Feature Selection` in the `Option` section.
      <details>
        <summary>example</summary> 
        <figure>
            <img src="image/feature.png">  
        </figure>
      </details>  

    - **Target / Solver**: Select a target column and a solver.
      - **Regression**: `Regression` solves regression problems. It uses models as `LGBMRegressor`, `XGBRegressor`, `RandomForestRegressor` and `ExtraTreesRegressor`.
        <details>
          <summary>example</summary>
          <figure>
              <img src="image/solver_regression.png">  
          </figure>  
        </details>

      - **Binary Classification**: `Binary Classification` solves classification problems. It uses models as `LGBMClassifier`, `XGBClassifier`, `RandomForestClassifier` and `ExtraTreesClassifier`.  Here, the target needs to be encoded. Set a percentage to separate the target by 0 and 1.
        <details>
          <summary>example</summary>
          <figure>
              <img src="image/solver_classification.png">  
          </figure>  
        </details>  

- **Check dataset**: You can check the prepared data set in real time.
  <details>
    <summary>example</summary>
    <figure>
        <img src="image/check_dataset.png">
    </figure>  
  </details>

### 2. Evaluation
When you click `Start` button, the app trains four models and select the best one automatically. For the regression models, the minimum `mae` (mean average error) model is selected. For the classification models, the maximum `auc` (area under the curve) model is selected. It provides visualization of prediction results and answers.  

<details>  
  <summary>example</summary>  

  |Regression|Binary Calssification|
  |:---:|:---:|
  |<img src="image/result_regression.png">|<img src="image/result_classification.png">|
</details>

### 3. Feature importance
In this section, you can see feature imporance for the best model. You can choose the number of features in the graph with the number input. When you click the `Download` button, you can download the feature importance data as a `csv` file.  

<details>  
  <summary>example</summary>  
  <figure>
      <img src="image/feature_importance.png">  
  </figure> 
</details>

### 4. Feature dependence
|SHAP|1D simulation|2D simulation|
|:---:|:---:|:---:|
|<img src="image/feature_dependence_shap.png" width="100%">|<img src="image/feature_dependence_1d.png" width="90%">|<img src="image/feature_dependence_2d.png" width="97%">|

- **SHAP**: SHAP dependence plot. SHAP means the contribution of features to predict the target. Please refer to the link below for more information on SHAP.
  - https://github.com/slundberg/shap
  - https://christophm.github.io/interpretable-ml-book/shap.html  
- **1D Simulation**: 1D simulation dependence plot. This shows the change in the target value for one selected feature. Other features are setted as mean value.
- **2D Simulation**: 2D simulation dependence plot. This shows the change in the target value for two selected features. Other features are setted as mean value.  
- **Optional, `Download`**: Download the shap/simulation data file.
- **Optional, `Extract all figures`**: Compresses the results for all features into a zip file. You can download the zip file with the download button that appears later.  

### 5. Prediction
You can make predictions about the test data. The test data should include features used for model training.  

<details>  
  <summary>example</summary>  
  <figure>
      <img src="image/prediction.png">  
  </figure> 
</details>  

- **`Make result file`**: Perform the prediction and make a result file. Generate a download button after the prediction is performed.
- **`Download`**: Download the result `csv` file.
  
## Optional
- **Auto feature selection**: `Auto feature selection` is an option to remove unimportant features automatically. Technically, it removes features whose importance lower than `random_noise`. `random_noise` is a feature that has random uniform distribution. Features with smaller `random_noise` importance can be considered insignificant.  

  <details>  
    <summary>example</summary>  
    <figure>
        <img src="image/auto_feature_selection.png">  
    </figure> 
  </details>

## Reference
1. SHAP: https://github.com/slundberg/shap
2. Titanic dataset: https://www.kaggle.com/c/titanic
3. Boston house dataset: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
