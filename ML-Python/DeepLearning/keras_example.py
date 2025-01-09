import keras
from keras.api.models import Sequential
from keras.api.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import requests as rq
import pandas as pd
import numpy as np

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
response = rq.get(url)
file_path = "./data.csv"

# First tested with Iris dataset
# Another dataset is about the concrete
"""
filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)
"""

if response.status_code == 200:
    with open(file_path, "w") as file:
        file.write(response.text)

df = pd.read_csv(file_path)
y = np.asanyarray(df["CO2EMISSIONS"])
X = np.asanyarray(df[["ENGINESIZE",'CYLINDERS']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

model = Sequential()
n_cols = X.shape[1]

model.add(Dense(4, activation='relu', input_shape=(n_cols,)))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train)

predict = model.predict(X_test)
print(y_test[:10])
print(predict[:10])
#print("Mean absolute error: %.2f" % np.mean(np.absolute(y_test - predict[1])))
