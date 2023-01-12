import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pickle


def train_test_splitting(df, target):
    
    X = df.drop(target, axis = 1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2,
                                                        stratify=y, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def normalization_data(norm, data):
    """
    on normalise la donnée rester dans une range de value 
    evite de biaiser le model
    """
    cols_to_scale = [column for column in data.columns]
    
    if norm==1:
        scale=StandardScaler()
    else:
        scale=MinMaxScaler()
        
    scale.fit(data[cols_to_scale])
    data[cols_to_scale] = scale.transform(data[cols_to_scale])
    
    return data


def model_training(model, X_train, y_train):
    
    model.fit(X_train,y_train)
    
    return model




    


    
