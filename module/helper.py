import streamlit as st
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
        if len(selected_values) > 0:
            df = pd.concat([df.loc[df[column]==value] for value in selected_values])
    return df

@st.cache(ttl=86400, allow_output_mutation=True)
def get_df_classification(df, target, l_q=20, h_q=80, random_state=42):
    # Target Values
    values = df[target].values
    
    # Get Low, High Values
    l_value = np.percentile(values, q=l_q)
    h_value = np.percentile(values, q=h_q)

    # Get Low, High Index
    l_index = np.where(values <= l_value)[0]
    h_index = np.where(values >= h_value)[0]

    # Get Low, High Data Number
    l_num = len(l_index)
    h_num = len(h_index)
    
    # Output
    l_df = df.iloc[l_index].copy()
    l_df[target] = 0
    h_df = df.iloc[h_index].copy()
    h_df[target] = 1
    df_out = pd.concat([l_df, h_df], axis=0)
    df_out = df_out.sample(frac=1, random_state=random_state)

    info = {}
    info['low'] = {'q': l_q, 'v': l_value, 'n': l_num}
    info['high'] = {'q': h_q, 'v': h_value, 'n': h_num}
    
    return df_out, info

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