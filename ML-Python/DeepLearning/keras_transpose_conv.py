import warnings
warnings.simplefilter('ignore')

import tensorflow as tf 
from keras.api.models import Model 
from keras.api.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, Dropout
import numpy as np 
import matplotlib.pyplot as plt 

input_layer = Input(shape=(28, 28, 1))

conv_layer = Conv2D(filters=32, kernel_size=(10, 10), activation='tanh', padding='same')(input_layer)
dropout_layer = Dropout(rate=0.5)(conv_layer)
transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(10, 10), activation='tanh', padding='same')(dropout_layer)

model = Model(inputs=input_layer, outputs=transpose_conv_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

# Generate synthetic training data 
X_train = np.random.rand(1000, 28, 28, 1) 
y_train = X_train # For reconstruction, the target is the input 
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

 # Generate synthetic test data 

X_test = np.random.rand(200, 28, 28, 1) 
y_test = X_test
loss = model.evaluate(X_test, y_test) 
print(f'Test loss: {loss}')

# Predict on test data 
y_pred = model.predict(X_test) 

# Plot some sample images 

n = 10 # Number of samples to display 

plt.figure(figsize=(20, 4))

for i in range(n): 

    # Display original 
    ax = plt.subplot(2, n, i + 1) 
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original") 
    plt.axis('off') 
    # Display reconstruction 
    ax = plt.subplot(2, n, i + 1 + n) 
    plt.imshow(y_pred[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show()