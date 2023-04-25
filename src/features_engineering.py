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



import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_subject(df):
    # Nettoyer les données
    cleaned_subjects = []
    for subject in df['subject']:
        subject = re.sub('[^a-zA-Z]', ' ', subject)
        subject = subject.lower()
        cleaned_subjects.append(subject)
    df['subject'] = cleaned_subjects

    # Lemmatisation et suppression des stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    lemmatized_subjects = []
    for subject in df['subject']:
        words = subject.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        lemmatized_subjects.append(' '.join(lemmatized_words))
    df['subject'] = lemmatized_subjects

    # Création de fonctionnalités
    subject_lengths = []
    num_words = []
    num_uppercase = []
    for subject in df['subject']:
        subject_lengths.append(len(subject))
        num_words.append(len(subject.split()))
        num_uppercase.append(sum(1 for c in subject if c.isupper()))
    df['subject_length'] = subject_lengths
    df['num_words'] = num_words
    df['num_uppercase'] = num_uppercase

    # Vectorisation
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['subject'])

    # Retourner la matrice de caractéristiques et le DataFrame mis à jour
    return X, df[['subject_length', 'num_words', 'num_uppercase']]

