import pandas as pd
# Regression Tree Algorithm
from sklearn.tree import DecisionTreeRegressor
# Split our data into a training and testing data
from sklearn.model_selection import train_test_split
import logging
import requests as rq

logging.basicConfig(level=logging.INFO)

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv"
file_path = "./crimeBoston.csv"

response = rq.get(url)

if response.status_code == 200:
    with open(file_path, "w") as file_conn:
        file_conn.write(response.text)

data = pd.read_csv(file_path)
data.dropna(inplace=True)
X = data.drop(columns=["MEDV"])
Y = data["MEDV"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)
regression_tree = DecisionTreeRegressor(criterion = 'absolute_error')
regression_tree.fit(X_train, Y_train)
regression_tree.score(X_test, Y_test)

prediction = regression_tree.predict(X_test)

print("$",(prediction - Y_test).abs().mean()*1000)

from sklearn.metrics import r2_score

r2_predict = r2_score(Y_test, prediction)
print(r2_predict)