import numpy as np
from pandas import DataFrame
from typing import Dict


def apply_table(x, table):
    if x in table:
        return table[x]
    else:
        return np.random.choice(range(len(table)), size=1)[0]

    
def encode(
     df: DataFrame,
     encoder: Dict[str, Dict[str, int]]
 ) -> DataFrame:
    for feature in encoder.keys():
        df[feature] = df[feature].apply(lambda x: apply_table(x, encoder[feature]))    
    return df
