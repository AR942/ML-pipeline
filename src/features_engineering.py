import pandas as pd
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer


def missing_values(df):
    """
    function to output a dataframe with sum of missing value per column
    """
    count = df.isnull().sum().values
    return pd.DataFrame(data={'column':df.columns.values, "na_sum": count})


def drop_nan30(df):
    """
    function to list the columns containing more than 30percent missing data
    and remove them from dataset
    """
    nan_cols30 = [column for column in df.columns if df[column].isnull().sum() > 0.30*len(df)]
    df.drop(columns = nan_cols30, inplace=True)
    return df

def dummies_creation(df):
    """
    listing des colonnes categorical and creation of dummies/one hot encoding
    """
    categorical_column = [col for col in df.columns if df[col].dtype == 'object']
    categorical_column

    dummies = pd.get_dummies(df[categorical_column], drop_first=True)
    dataset_dummies = pd.concat([df, dummies], axis='columns')

    dataset_dummies.drop(columns = categorical_column, axis = 1, inplace = True)
    
    return dataset_dummies

    
def remove_nan(df):
    """
    function to replace the remaining nans values with the median 
    of each numerical column en utilisant simple imputer
    """
    imputer = SimpleImputer(strategy ="median")
    imputer.fit(df)
    dataset_clean = imputer.transform(df)
    
    dataset_clean = pd.DataFrame(dataset_clean, columns= df.columns)

    return dataset_clean

def convert_column(column : pd.Series, type):
    """
    just to remove the float from our target column...
    not really necessary i know
    """
    column = column.astype(type)
    return column



def preprocess_text(text):
    #enlever la poncutation
    text = text.translate(str.maketrans('', '', string.punctuation))

    #lowercase pour facilitation
    text = text.lower()

    #traitement des valeurs textuelles non voulues
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)


    text = text.replace('[^\w\s]', '')
    text = text.replace('\r', '')
    text = text.replace('\n', '')
    
    #tokenisation
    tokens = word_tokenize(text)

    #enlever les stopwerds anglais
    """stopwords = set(stopwords.words('english'))
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))"""
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    
    #Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    #Racinisation
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]


    #remettre sous forme de string
    text = ' '.join(tokens)

    return text

df.Synopsis = df.Synopsis.apply(preprocess_text)
