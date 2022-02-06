import numpy as np
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from typing import Tuple, Dict


def encode_category(
    df: DataFrame
) -> Tuple[DataFrame, Dict[str, int]]:
    category_columns = [column for column in df.columns if not is_numeric_dtype(df[column])]
    encoder = {}
    for column in category_columns:
        uniques = np.sort(np.unique(df[column].values))
        encoder[column] = {value:i for i, value in enumerate(uniques)}
        df[column] = df[column].apply(lambda x: encoder[column][x])
    return df, encoder
