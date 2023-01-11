
DataSet (Finance use case)
DataSet of Home Credit Risk Classification: https://www.kaggle.com/c/home-credit-default-risk/data
you'll not use all the datasets available on Kaggle, only the main data set:⇒application_train.csv⇒application_test.csv
You may also use a reduced version of these datasets

download these datasets, rename application_train.csv to train.csv
and rename application_test.csv to test.csv
put them in classification-risk-app/data folder


first : cd path/to/risk-classification-app
<<<<<<< HEAD
then : pip install -r requirements.txt
=======
then : 
1 - create the venv for this project
python -m venv .venvarthus
2 - activate your conda environement
source .venvarthus/bin/activate
3- install packages requirements for the project inside your venv
pip install -r requirements.txt

>>>>>>> a468811 (another commit)
then in your terminal run mlflow ui
troubleshoot : if you get mflow connexion error run this on your terminal 
pkill -f gunicorn
and then run mlflow ui again

<<<<<<< HEAD
then cd src folder in risk classification app and run:
=======
then 
cd src folder in risk classification app 
and run:
>>>>>>> a468811 (another commit)
python main.py
