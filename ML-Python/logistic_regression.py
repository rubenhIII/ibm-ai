import logging
import requests as rq
import os.path

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
file_path = "./ChurnData.csv"

if os.path.isfile(file_path):
    logging.info("File already exists!")
else:
    logging.info("Downloading file ...")
    response = rq.get(url)
    if response.status_code == 200:
        with open(file_path, "w") as file_conn:
            file_conn.write(response.text)

churn_df = pd.read_csv(file_path)

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
logging.info(f'{churn_df.columns} {churn_df.shape}')

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)

from sklearn.metrics import jaccard_score, classification_report
print(jaccard_score(y_test, yhat,pos_label=0))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, yhat)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

print(classification_report(y_test, yhat))
'''
Based on the count of each section, we can calculate precision and recall of each label:

Precision is a measure of the accuracy provided that a class label has been predicted.
It is defined by: precision = TP / (TP + FP)

Recall is the true positive rate. It is defined as: Recall =  TP / (TP + FN)

So, we can calculate the precision and recall of each class.

F1 score: Now we are in the position to calculate the F1 scores for each label based on the 
precision and recall of that label.

The F1 score is the harmonic average of the precision and recall, where an F1 score reaches 
its best value at 1 (perfect precision and recall) and worst at 0. 
It is a good way to show that a classifer has a good value for both recall and precision.
'''

from sklearn.metrics import log_loss
print(log_loss(y_test, yhat_prob))
#Log loss( Logarithmic loss) measures the performance of a 
#classifier where the predicted output is a probability value between 0 and 1.