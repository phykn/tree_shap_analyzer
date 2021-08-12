import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

@st.cache()
def load_data(file_path, max_len=100000, random_state=42, add_random_column=True):
    # Load Data
    df = pd.read_csv(file_path)
    df = df.sample(n=max_len, random_state=random_state) if len(df) > max_len else df

    # Select numeric column
    columns = [column for column in df.columns if is_numeric_dtype(df[column])]
    df = df[columns]

    # Add random column
    if add_random_column:        
        np.random.seed(random_state)
        df['random_noise'] = np.random.rand(len(df))
    return df

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