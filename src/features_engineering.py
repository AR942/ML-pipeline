import pandas as pd
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""from data_preparation import dataset"""

def missing_values(df):
    count = df.isnull().sum().values
    return pd.DataFrame(data={'column':df.columns.values, "na_sum": count})


def drop_nan30(df):
    nan_cols30 = [column for column in df.columns if df[column].isnull().sum() > 0.30*len(df)]
    df.drop(columns = nan_cols30, inplace=True)
    return df

def dummies_creation(df):
    
    categorical_column = [col for col in df.columns if df[col].dtype == 'object']
    categorical_column

    dummies = pd.get_dummies(df[categorical_column], drop_first=True)
    dataset_dummies = pd.concat([df, dummies], axis='columns')

    dataset_dummies.drop(columns = categorical_column, axis = 1, inplace = True)
    
    return dataset_dummies

    
def remove_nan(df):
    
    imputer = SimpleImputer(strategy ="median")
    imputer.fit(df)
    dataset_clean = imputer.transform(df)
    
    dataset_clean = pd.DataFrame(dataset_clean, columns= df.columns)

    return dataset_clean

def convert_column(column : pd.Series, type):
    column = column.astype(type)
    return column

"""nan = missing_values(dataset).sort_values(["na_sum"], ascending=False)
nan

dataset = drop_nan30(dataset)
dataset

dataset_dummies = dummies_creation(dataset)
dataset_dummies

dataset_clean = remove_nan(dataset_dummies)
dataset_clean

dataset_clean["TARGET"] = convert_column(dataset_clean["TARGET"], int)


print(dataset_clean.shape)"""
