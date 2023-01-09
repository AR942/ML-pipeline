import pandas as pd
import datetime
datetime.datetime.now()
import mlflow
from sklearn.metrics import accuracy_score,precision_score,recall_score

"""from predict import get_metrics, predictions, run_metrics, classifier"""


def create_experiment(experiment_name,run_name, run_metrics, model, confusion_matrix_path, run_params=None):
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name) 
    """create experiment"""
    
    with mlflow.start_run(run_name=run_name):
        
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
            
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])
        
        
        if not confusion_matrix_path == None:
            mlflow.log_artifact(confusion_matrix_path, 'confusion_matrix')
        
        """tag the experiment"""
        mlflow.set_tag("tag1", "Random Forest")
        
        """record the model"""
        mlflow.sklearn.log_model(model, "model")
        

            
    print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))
    
    
"""date =  datetime.datetime.now().strftime("%Hh%M_%d-%m-%Y")   
experiment_name = "RDF_classifier"+ date
run_name="RDF_classifier"+date
create_experiment(experiment_name, run_name, run_metrics, classifier, '../output/confusion_matrix.png' )"""
