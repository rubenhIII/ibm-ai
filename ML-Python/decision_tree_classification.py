import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import requests as rq
import pandas as pd
import numpy as np
import logging
import os.path

from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

logging.basicConfig(level=logging.INFO)

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
file_path = "./drug200.csv"

response = rq.get(url)

if response.status_code == 200:
    with open(file_path, "w") as file_conn:
        file_conn.write(response.text)

my_data = pd.read_csv(file_path)
logging.info(f'{my_data.info}')

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Changing categorical values to numerical.
# Scikit learn Decision Tree algorithm allows only numerical values.

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

y = my_data["Drug"]

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
logging.info(f'X train shape: {X_trainset.shape} X test shape: {X_testset.shape}')

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
drugTree.fit(X_trainset,y_trainset)
predTree = drugTree.predict(X_testset)

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

tree.plot_tree(drugTree)
plt.show()