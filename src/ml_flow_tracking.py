import pandas as pd
import datetime
datetime.datetime.now()
import mlflow
from sklearn.metrics import accuracy_score,precision_score,recall_score

def create_experiment(experiment_name,run_name, run_metrics, model, confusion_matrix_path):
    """
    function pour créer l'experiment mlflow pour pouvoir sauvegarder nos experiences de modélisation
    """
    
    
    """set the localhost url where experiments of mlflow will be hosted"""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
     
    """create experiment"""
    with mlflow.start_run(run_name=run_name):
        
            
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])
        
        """log classification object inside mlflow exp"""
        if not confusion_matrix_path == None:
            mlflow.log_artifact(confusion_matrix_path, 'confusion_matrix')
        
        """tag the experiment"""
        mlflow.set_tag("tag1", "Random Forest")
        
        """record the model"""
        mlflow.sklearn.log_model(model, "model")
        
    print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))
    
    

