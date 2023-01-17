
DataSet (Finance use case)
DataSet of Home Credit Risk Classification: https://www.kaggle.com/c/home-credit-default-risk/data
you'll not use all the datasets available on Kaggle, only the main data set:⇒application_train.csv⇒application_test.csv
You may also use a reduced version of these datasets

download these datasets

put them in risk-classification/data folder
Note :  python used for this project is 3.10.8

first : 

>cd path/to/risk-classification

then : pip install virtualenv

1 - create the virtual environement for this project (first pip install virtualenv)

>python -m venv .venvarthus

2 - activate your virtual environement and choose latest python interpreter on vscode or pycharme for exemple

>source .venvarthus/bin/activate

3- install packages requirements for the project inside your venv

>pip install -r requirements.txt

then in your terminal run 

>mlflow ui --host http://127.0.0.1:5000

troubleshoot : if you get mflow connexion error run this on your terminal 

>pkill -f gunicorn

and then run 

>mlflow ui --host http://127.0.0.1:5000

then 

>cd src folder in risk-classification app 

and run:

>python main.py


## Pour les visualisations avec shap, voici le [notebook](notebook/notebook-avec-shap.ipynb)
# Les images des visualisations shap sont disponibles [ici](shap-plots)





### Full process run of the ML pipeline with pictures : 

# Creation du venv et start du projet

![image](https://i.postimg.cc/kGpwbN7k/Capture-d-e-cran-2023-01-17-a-17-01-56.png)

![image](https://i.postimg.cc/257wwNzd/Capture-d-e-cran-2023-01-17-a-17-03-35.png)


# Run la modélisation train/test

![image](https://i.postimg.cc/LXy3ckgn/Capture-d-e-cran-2023-01-17-a-17-12-06.png)


# Run l'experiment MLFLOW

![image](https://i.postimg.cc/0jb0P67Z/Capture-d-e-cran-2023-01-17-a-17-17-39-min.png)

![image](https://i.postimg.cc/BQPBzKSn/Capture-d-e-cran-2023-01-17-a-17-17-55-min.png)

# Test du modèle avec les données test de application_test.csv
![image](https://i.postimg.cc/L5wBXHTS/Capture-d-e-cran-2023-01-17-a-17-18-08-min.png)

![image](https://i.postimg.cc/8CS4MzVP/Capture-d-e-cran-2023-01-17-a-17-19-27-min.png)

# Check de l'experiment MLFLOW sur le localhost http://127.0.0.1:5000
![image](https://i.postimg.cc/vZRtwZTL/Capture-d-e-cran-2023-01-17-a-17-20-00-min.png)

# visualisation du csv de l'export des prédictions sur données test
![image](https://i.postimg.cc/4NcxqJrg/Capture-d-e-cran-2023-01-17-a-17-19-04-min.png)


