from model_training import test_normalized, y_test
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
import datetime
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score

def predict(model, data):
    
    predictions = model.predict(data)
    return  predictions

def get_metrics(y_test, predictions):
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    return {'accuracy': round(acc, 2), 
            'precision': round(prec, 2), 
            'recall': round(recall, 2)}


classifier =  pickle.load(open('../model/RDF_classifier.pkl', 'rb'))
predictions = predict(classifier, test_normalized)
get_metrics(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
print(cm)

def export_prediction(data, predict):

    date =  datetime.datetime.now().strftime("%Hh%M_%d-%m-%Y")
    path = "../output"

    index = data.index
    prediction_final = pd.DataFrame({"ID": index, "prediction_label": predict})

    prediction_final.to_csv(os.path.join(path, "predictions-"+ date +".csv"))

export_prediction(test_normalized, predictions)