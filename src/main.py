import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pickle
import datetime
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from make_dataset import get_dataframe, index_setting
from data_preparation import map_function
from features_engineering import remove_nan, dummies_creation, convert_column
from model_training import train_test_splitting, normalization_data, model_training
from predict import predict, get_metrics, create_confusion_matrix_plot, export_prediction
from ml_flow_tracking import create_experiment 

while True:
    modelisation = str(input("Do you want to modelise a train or a test ? :\n"))
    try:
        assert modelisation in ["train", "test"]
        print(f'You choosed to {modelisation} the model')
        break
    except AssertionError:
        pass

if modelisation == 'train':
    
    try:
        dataset = get_dataframe("train")
    except Exception as e:
        logging.exception(
            "No data to load : check file path"
        )
    
    index_setting(dataset, "SK_ID_CURR")
    
    dataset.NAME_TYPE_SUITE = map_function(dataset.NAME_TYPE_SUITE, 5000, "other_NAME_TYPE_SUITE")
    dataset.NAME_INCOME_TYPE = map_function(dataset.NAME_INCOME_TYPE, 22000, "other_NAME_INCOME_TYPE")
    dataset.NAME_EDUCATION_TYPE = map_function(dataset.NAME_EDUCATION_TYPE, 5000, "other_NAME_EDUCATION_TYPE")
    dataset.NAME_FAMILY_STATUS = map_function(dataset.NAME_FAMILY_STATUS, 17000, "other_NAME_FAMILY_STATUS")
    dataset.NAME_HOUSING_TYPE = map_function(dataset.NAME_HOUSING_TYPE, 10000, "other_NAME_HOUSING_TYPE")
    dataset.ORGANIZATION_TYPE = map_function(dataset.ORGANIZATION_TYPE, 2500, "other_ORGANIZATION_TYPE")
    dataset.ORGANIZATION_TYPE = map_function(dataset.ORGANIZATION_TYPE, 9000, "other2_ORGANIZATION_TYPE")
    
    dataset_dummies = dummies_creation(dataset)

    dataset_clean = remove_nan(dataset_dummies)

    dataset_clean["TARGET"] = convert_column(dataset_clean["TARGET"], int)
    print(dataset_clean.shape)
    
 
    X_train, X_test, y_train, y_test = train_test_splitting(dataset_clean, "TARGET")

    dataset_clean.shape

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    
    train_normalized = normalization_data(1, X_train)
    test_normalized = normalization_data(1, X_test)
    
    """
    inbalance target class distribution : using undersampling method
    """
    rus = RandomUnderSampler() 
    # undersampling train_normalized, y_train
    X_rus, y_rus = rus.fit_resample(train_normalized, y_train)
    # new TARGET class distribution
    print(Counter(y_rus))
    
    classifier = model_training(RandomForestClassifier(), X_rus, y_rus)
    pickle.dump(classifier, open("../model/RDF_classifier.pkl", 'wb'))
    
    date =  datetime.datetime.now().strftime("%Hh%M_%d-%m-%Y")
    classifier =  pickle.load(open('../model/RDF_classifier.pkl', 'rb'))
    predictions = predict(classifier, test_normalized)
    run_metrics = get_metrics(y_test, predictions)
    print(run_metrics)
    create_confusion_matrix_plot(classifier, y_test, predictions, '../output/confusion_matrix.png')

    export_prediction(test_normalized, predictions, "../output")
    
    date =  datetime.datetime.now().strftime("%Hh%M_%d-%m-%Y")   
    experiment_name = "RDF_classifier"+ date
    run_name="RDF_classifier"+date
    create_experiment(experiment_name, run_name, run_metrics, classifier, '../output/confusion_matrix.png' )

else:
    print("no test atm")
    
    

"""import os
  
os.system('python ./make_dataset.py')
os.system('python ./data_preparation.py')
os.system('python ./features_engineering.py')
os.system('python ./model_training.py')
os.system('python ./predict.py')
os.system('python ./ml_flow_tracking.py')"""
