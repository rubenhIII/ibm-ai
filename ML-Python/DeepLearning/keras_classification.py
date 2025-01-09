from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets

import requests as rq
import pandas as pd
import numpy as np

"""
# import the data
from keras.datasets import mnist

# read the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

Notes:
With conventional neural networks, we cannot feed in the image as input as is. 
So we need to flatten the images into one-dimensional vectors, 
each of size 1 x (28 x 28) = 1 x 784.

# flatten images into one-dimensional vector

num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

"""

iris = datasets.load_iris(as_frame=True)
df_x, df_y = datasets.load_iris(return_X_y=True, as_frame=True)
data = iris.frame

print(df_y)
X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)']].values
y = data['target']

print(X[0:10])
print(y[0:10])

import warnings
warnings.simplefilter('ignore', FutureWarning)
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

model = Sequential()
n_cols = X.shape[1]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model.add(Dense(3, activation='relu', input_shape=(n_cols,)))
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10)
predict = model.predict(X_test)

scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))

# model.save('classification_model.keras')
# pretrained_model = keras.saving.load_model('classification_model.keras')
