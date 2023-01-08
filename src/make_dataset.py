import pandas as pd
from glob import glob

# --------------------------------------------------------------
# List all data in data
# --------------------------------------------------------------
files = glob("../data/*csv")
files


def get_dataframe(data):
    if data == "train": 
        df = pd.read_csv("../data/train.csv")
        return df
    else:
        df = pd.read_csv("../data/test.csv")
        return df
    
# --------------------------------------------------------------
# define new index
# --------------------------------------------------------------
def index_setting(df, index):
    return df.set_index(index, inplace=True)

