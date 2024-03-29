import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional


def read_csv(
    path: str,
    max_len: Optional[int]=None,
    add_random_noise: bool=False,
    random_state: int=42    
) -> DataFrame:
    """read a data file.
    
    path: The path of the data file.
    max_len: Maximum data size of the output dataframe.
    add_random_noise: When it is True, a column is added to the output dataframe that is composed of random uniform numbers (0, 1).
                      Random noise will be a baseline for feature importance.
    random_state: Setting for randomness.
    """

    # load data
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except:
        df = pd.read_csv(path, encoding="cp949")

    # set maximum output size
    index = np.array(df.index)
    if max_len:
        if len(index) > max_len:
            np.random.seed(random_state)
            index = np.random.choice(index, size=max_len, replace=False)
            index = np.sort(index)
    df = df.loc[index].reset_index(drop=True)

    # add random column
    if add_random_noise:        
        np.random.seed(random_state)
        df["random_noise"] = np.random.uniform(low=0.0, high=1.0, size=len(df))

    return df