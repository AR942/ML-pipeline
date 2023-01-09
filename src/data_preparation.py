import pandas as pd
import numpy as np
import logging

import make_dataset
from make_dataset import get_dataframe, index_setting
# --------------------------------------------------------------
# Get dataframe
# --------------------------------------------------------------
"""try:
    dataset = get_dataframe("train")
except Exception as e:
    logging.exception(
        "No data to load : check file path"
    )
dataset.TARGET.value_counts()
dataset"""
"""dataset = pd.read_csv("risk-classification/data/dataset.csv")
"""
# --------------------------------------------------------------
# define new index
# --------------------------------------------------------------
"""index_setting(dataset, "SK_ID_CURR")
dataset
"""
def map_function(column: pd.Series, threshold: int, target: str):

    count = column.value_counts()
    rares = count.loc[count < threshold].keys()
    new_serie = column.copy()
    
    for value in rares:
        new_serie.replace(value, target, inplace=True)
    return new_serie

"""dataset.NAME_TYPE_SUITE = map_function(dataset.NAME_TYPE_SUITE, 5000, "other_NAME_TYPE_SUITE")
dataset.NAME_INCOME_TYPE = map_function(dataset.NAME_INCOME_TYPE, 22000, "other_NAME_INCOME_TYPE")
dataset.NAME_EDUCATION_TYPE = map_function(dataset.NAME_EDUCATION_TYPE, 5000, "other_NAME_EDUCATION_TYPE")
dataset.NAME_FAMILY_STATUS = map_function(dataset.NAME_FAMILY_STATUS, 17000, "other_NAME_FAMILY_STATUS")
dataset.NAME_HOUSING_TYPE = map_function(dataset.NAME_HOUSING_TYPE, 10000, "other_NAME_HOUSING_TYPE")
dataset.ORGANIZATION_TYPE = map_function(dataset.ORGANIZATION_TYPE, 2500, "other_ORGANIZATION_TYPE")
dataset.ORGANIZATION_TYPE = map_function(dataset.ORGANIZATION_TYPE, 9000, "other2_ORGANIZATION_TYPE")


dataset"""







