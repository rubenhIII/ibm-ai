import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import requests as rq
import pandas as pd
import numpy as np
import logging


logging.basicConfig(level=logging.INFO)

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
response = rq.get(url)
file_path = "./data.csv"

if response.status_code == 200:
    with open(file_path, "w") as file:
        file.write(response.text)

df = pd.read_csv(file_path)
logging.info(df.describe())

viz = df[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

plt.scatter(df.FUELCONSUMPTION_COMB, df.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(df.CYLINDERS, df.CO2EMISSIONS,  color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
logging.info(f'Coefficients: {regr.coef_}')
logging.info(f'Intercept: {regr.intercept_}')

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

logging.info('Training with ENGINESIZE')
logging.info("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
logging.info("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
logging.info("R2-score: %.2f" % r2_score(test_y , test_y_))

# Training with FUELCONSUMPTION_COMB
train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
test_y_ = regr.predict(test_x)
logging.info('Training with FUELCONSUMPTION_COMB')
logging.info("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
logging.info("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
logging.info("R2-score: %.2f" % r2_score(test_y , test_y_))