# Projet de classification binaire de risques financiers avec la création d'une pipeline ML en utilisant MLFLOW et Shap dans 

#### Groupe DE2
#### Arthus ROUHI
#### Adam SOUSSOU
#### Lucas BENYAMIN


## Les datasets requis pour ce projet se trouvent sur ce lien : https://www.kaggle.com/c/home-credit-default-risk/data
### Seuls les datasets application_train.csv et application_test.csv seront utilisés dans ce projet



# Marche à suivre pour lancer la pipeline ML

download these 2 datasets

put them in risk-classification/data folder

### Note :  La version python utilisée dans ce projet est Python 3.10.8 
#### Tous les scripts de la pipeline ML se trouvent [ici](src)

first : 

>cd path/to/risk-classification

then : pip install virtualenv

1 - create the virtual environement for this project (first pip install virtualenv)

>python -m venv .venvarthus

2 - activate your virtual environement and choose latest python interpreter on vscode or pycharme for exemple

>source .venvarthus/bin/activate

3- install packages requirements for the project inside your venv

>pip install -r requirements.txt

then in your terminal run mlflow (host HAS To be http://127.0.0.1:5000/)

>mlflow ui

Troubleshoot : if you get mflow connexion error run this on your terminal 

>pkill -f gunicorn

and then run 

>mlflow ui --host http://127.0.0.1:5000

4- Run the ML Pipeline

Go to the src folder inside the project

>cd risk-classification/src

Run the file main.py:

>python main.py


# Pour les visualisations avec shap, voici le [notebook](notebook/notebook-avec-shap.ipynb)
## Les images des visualisations shap sont disponibles [ici](shap-plots)





# Full process run of the ML pipeline with pictures : 

## Creation du venv et start du projet

![image](https://i.postimg.cc/kGpwbN7k/Capture-d-e-cran-2023-01-17-a-17-01-56.png)

![image](https://i.postimg.cc/257wwNzd/Capture-d-e-cran-2023-01-17-a-17-03-35.png)


## Run la modélisation train/test

![image](https://i.postimg.cc/LXy3ckgn/Capture-d-e-cran-2023-01-17-a-17-12-06.png)


## Run l'experiment MLFLOW

![image](https://i.postimg.cc/0jb0P67Z/Capture-d-e-cran-2023-01-17-a-17-17-39-min.png)

![image](https://i.postimg.cc/BQPBzKSn/Capture-d-e-cran-2023-01-17-a-17-17-55-min.png)

## Test du modèle avec les données test de application_test.csv
![image](https://i.postimg.cc/L5wBXHTS/Capture-d-e-cran-2023-01-17-a-17-18-08-min.png)

![image](https://i.postimg.cc/8CS4MzVP/Capture-d-e-cran-2023-01-17-a-17-19-27-min.png)

## Check de l'experiment MLFLOW sur le localhost http://127.0.0.1:5000
![image](https://i.postimg.cc/vZRtwZTL/Capture-d-e-cran-2023-01-17-a-17-20-00-min.png)

## visualisation du csv de l'export des prédictions sur données test
![image](https://i.postimg.cc/4NcxqJrg/Capture-d-e-cran-2023-01-17-a-17-19-04-min.png)


