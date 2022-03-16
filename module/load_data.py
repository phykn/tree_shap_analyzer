import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

@st.cache(ttl=86400, allow_output_mutation=True)
def load_data(file_path, max_len=100000, random_state=42, add_random_column=True):
    # Load Data
    try:
        df = pd.read_csv(file_path, encoding='cp949')
    except:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            except:
                raise ValueError('File Load Error.')

    # Sample in maximum length
    df = df.sample(n=max_len, random_state=random_state) if len(df) > max_len else df

    # Add random column
    if add_random_column:        
        np.random.seed(random_state)
        df['random_noise'] = np.random.rand(len(df))

    # Get numeric column
    numeric_columns = [column for column in df.columns if is_numeric_dtype(df[column])]
    return df, numeric_columns

@st.cache(ttl=86400, allow_output_mutation=True)
def del_outlier(df, features, lower_limit=0.01, upper_limit=0.99):
    df_out = df.reset_index(drop=True)
    lower_limits = df.quantile(q=lower_limit)
    upper_limits = df.quantile(q=upper_limit)
    for column in features:
        values = df[column].values.astype(float)
        index  = np.where(np.logical_and(values > lower_limits[column], 
                                         values < upper_limits[column]))
        df_out = df_out.loc[index]
    return df_out

@st.cache(ttl=86400, allow_output_mutation=True)
def nan_process(df, nan_method='Delete'):
    nan_methods = ['Delete', 'Mean', 'Median']
    assert nan_method in nan_methods, f'nan_method not in {nan_methods}.'

    if nan_method == 'Delete':
        return df.dropna()
    elif nan_method == 'Mean':
        return df.fillna(df.mean())
    elif nan_method == 'Median':
        return df.fillna(df.median())