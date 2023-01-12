import pandas as pd

def get_dataframe(data):
    
    if data == "train": 
        df = pd.read_csv("../data/application_train.csv")
        return df
    else:
        df = pd.read_csv("../data/application_test.csv")
        return df
    
# --------------------------------------------------------------
# define new index
# --------------------------------------------------------------
def index_setting(df, index):
    return df.set_index(index, inplace=True)

