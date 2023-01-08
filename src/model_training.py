import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pickle

from features_engineering import dataset_clean

dataset_clean

def train_test_splitting(df, target):
    
    X = df.drop(target, axis = 1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2,
                                                        stratify=y, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_splitting(dataset_clean, "TARGET")

dataset_clean.shape

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

def normalization_data(norm, data):
    
    cols_to_scale = [column for column in data.columns]
    
    if norm==1:
        scale=StandardScaler()
    else:
        scale=MinMaxScaler()
        
    scale.fit(data[cols_to_scale])
    data[cols_to_scale] = scale.transform(data[cols_to_scale])
    
    return data

train_normalized = normalization_data(1, X_train)
test_normalized = normalization_data(1, X_test)

train_normalized

def model_training(model, X_train, y_train):
    
    model.fit(X_train,y_train)
    
    return model


train = True

if train:
    classifier = model_training(RandomForestClassifier(), train_normalized, y_train)
    pickle.dump(classifier, open("../model/RDF_classifier.pkl", 'wb'))


"""loaded_model = pickle.load(open(filename, 'rb'))"""
    

"""def predict(model, data, y_test):
    
    predictions = model.predict(data)
    cm = confusion_matrix(y_test, predictions)
    return cm, predictions

cm, predictions = predict(RDF_classifier, test_normalized, y_test)

index = pd.DataFrame(test_normalized.index)
predictions = pd.DataFrame(predictions)
predictions = pd.concat([index, predictions], axis=1)

submission = predictions.to_csv('/Users/arthusrouhi/Desktop/risk-classification/data/predictions.csv')

cm"""
    
    
