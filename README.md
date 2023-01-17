
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



