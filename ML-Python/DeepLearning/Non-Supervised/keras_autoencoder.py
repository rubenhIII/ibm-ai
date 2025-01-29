import tensorflow as tf
import numpy as np
from keras.api.layers import Input, Dense
from keras.api.models import Model
from keras.api.datasets.mnist import load_data

# Define encoder
input_layer = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_layer)

# Bottleneck
bottleneck = Dense(32, activation='relu')(encoded)

# Define decoder
decoded = Dense(64, activation='relu')(encoded)
output_layer = Dense(784, activation='sigmoid')(decoded)

# Combine the encoder and decoder into an autoencoder model
autoencoder = Model(input_layer, output_layer)
# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
print(autoencoder.summary())

(x_train, _), (x_test, _) = load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_train = x_train.reshape((len(x_test), np.prod(x_train.shape[1:])))

# Train the autoencoder
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Fine-tuning the autoencoder
# Unfreeze the top layers of the encoder
for layer in autoencoder.layers[-4:]:
    layer.trinable = True
# Compile the model again
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# Train the model again
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
