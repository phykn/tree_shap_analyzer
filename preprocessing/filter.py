from pandas import DataFrame
from typing import Dict, List, Any


def apply_filter(
    df: DataFrame,
    filter: Dict[str, List[Any]]
) -> DataFrame:
    '''Apply filter to DataFrame'''
    
    df = df.copy()

    if len(filter) > 0:
        for column, values in filter.items():
            df = df.loc[df[column].isin(values)]

    df = df.reset_index(drop=True)
    return df
