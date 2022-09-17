import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from typing import Tuple, Dict


def encode(
        x: str,
        encoder: Dict[str, float]
    ) -> float:
        if pd.isna(x):
            return np.nan
        else:
            return encoder[x]


def encode_category(
    df: DataFrame
) -> Tuple[DataFrame, Dict[str, int]]:
    category_columns = [column for column in df.columns if not is_numeric_dtype(df[column])]
    encoder = {}
    for column in category_columns:
        uniques = np.sort(np.unique(df[column].dropna().values))
        encoder[column] = {value:i for i, value in enumerate(uniques)}
        df[column] = df[column].apply(lambda x: encode(x=x, encoder=encoder[column]))
    return df, encoder
