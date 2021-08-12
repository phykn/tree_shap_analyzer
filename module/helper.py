import io
import base64
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import savgol_filter

def get_df_filter(df, name_1='', name_2=''):
    column = st.selectbox(name_1, [None]+list(df.columns))
    if column is not None:
        values = list(np.sort(df[column].dropna().unique()))
        selected_values = st.multiselect(name_2,
                                         options = values,
                                         default = values)
        if len(selected_values) > 1:
            df = pd.concat([df.loc[df[column]==value] for value in selected_values])
    return df

def Smoothing(x, y, window_length=9, polyorder=2):
    assert len(x)==len(y), 'Error. Not same length.'
    index = np.argsort(x)
    x = x[index]
    y = y[index]
    y_length = len(y)    
    if y_length <= polyorder:
        out = np.full_like(y, np.mean(y))
    else:
        if y_length < window_length:
            if y_length % 2 == 0:
                window_length = y_length - 1
            else:
                window_length = y_length
                
        if window_length <= polyorder:
            out = np.full_like(y, np.mean(y))
        else:
            out = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    return x, out

def get_table_download_link(df, file_name='file.csv', title='Download'):
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">{title}</a>'
    return href

def get_figure_download_linkzip_b64(figs, file_name='file.csv', title='Download'):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
        for i, fig in enumerate(figs):
            buf = io.BytesIO()
            fig.savefig(buf, bbox_inches='tight')
            name = f'fig_{i:03d}.png'
            zf.writestr(name, buf.getvalue())

    b64 = base64.b64encode(zip_buffer.getvalue()).decode()
    href = f'<a href="data:file/zip;base64,{b64}" download="{file_name}">{title}</a>'
    return href