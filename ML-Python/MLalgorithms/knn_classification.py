import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import requests as rq
import pandas as pd
import numpy as np
import logging

from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO)

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
response = rq.get(url)
file_path = "./teleCust.csv"

if response.status_code == 200:
    with open(file_path, "w") as file_conn:
        file_conn.write(response.text)

df = pd.read_csv(file_path)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
y = df['custcat'].values

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
logging.info(f'Train set: {X_train.shape} {y_train.shape}')
logging.info(f'Test set: {X_test.shape} {y_test.shape}')

k = 6
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(X_test)

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))