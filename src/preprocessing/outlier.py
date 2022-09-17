import numpy as np
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype


def delete_outlier(
    df: DataFrame,
    lower_limit: float=0.01, 
    upper_limit: float=0.99,
    random_noise_name: str="random_noise"
) -> DataFrame:
    """Select data between the lower_limit (percentage) and the upper_limit (percentage)"""

    numeric_columns = [column for column in df.columns if is_numeric_dtype(df[column])]
    numeric_columns = [column for column in numeric_columns if column != random_noise_name]

    lower_limits = df[numeric_columns].quantile(q=lower_limit)
    upper_limits = df[numeric_columns].quantile(q=upper_limit)

    for column in numeric_columns:
        values = df[column].values.astype(float)
        index = np.where(
            np.logical_and(values>=lower_limits[column], values<=upper_limits[column])
        )
        df = df.iloc[index]

    return df