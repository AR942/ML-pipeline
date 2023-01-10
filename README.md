
DataSet (Finance use case)
DataSet of Home Credit Risk Classification: https://www.kaggle.com/c/home-credit-default-risk/data
you'll not use all the datasets available on Kaggle, only the main data set:⇒application_train.csv⇒application_test.csv
You may also use a reduced version of these datasets

download these datasets, rename application_train.csv to train.csv
and rename application_test.csv to test.csv
put them in classification-risk-app/data folder


first : cd path/to/risk-classification-app
then : pip install -r requirements.txt
then in your terminal run mlflow ui
troubleshoot : if you get mflow connexion error run this on your terminal 
pkill -f gunicorn
and then run mlflow ui again

then cd src folder in risk classification app and run:
python main.py
