from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import datetime
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score, classification_report

def predict(model, data):
    """
    faire des predictions avec le modele
    """
    predictions = model.predict(data)
    return  predictions

def get_metrics(y_test, predictions):
    """
    évaluer les modeles avec des metrics de base de classification
    """
    
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    
    return {'accuracy': round(acc, 2), 
            'precision': round(prec, 2), 
            'recall': round(recall, 2)}

def create_confusion_matrix_plot(model, y_test, predictions, path):
    """
    créer et export la matrice de confusion
    """
    
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(path)

def export_prediction(data, predict, path):
    """
    export des prédictions du modele en csv sur les données test
    """

    date =  datetime.datetime.now().strftime("%Hh%M_%d-%m-%Y")

    index = data.index
    prediction_final = pd.DataFrame({"ID": index, "prediction_label": predict})

    prediction_final.to_csv(os.path.join(path, "predictions-"+ date +".csv"))



