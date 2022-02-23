import numpy as np
from pandas import DataFrame


def clear_data(
    df: DataFrame
) -> DataFrame:
    '''Select useable columns'''

    columns = []
    for column in df.columns:
        unique = np.unique(df[column].dropna())
        if len(unique) > 1:
            columns.append(column)

    return df[columns]
