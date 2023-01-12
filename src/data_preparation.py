import pandas as pd
import numpy as np
import logging

import make_dataset
from make_dataset import get_dataframe, index_setting

def map_function(column: pd.Series, threshold: int, target: str):
    """ function to bucketize outliers inside a column

    Returns:
        a column with buckétized value
    """
    count = column.value_counts()
    rares = count.loc[count < threshold].keys()
    new_serie = column.copy()
    
    for value in rares:
        new_serie.replace(value, target, inplace=True)
    return new_serie








