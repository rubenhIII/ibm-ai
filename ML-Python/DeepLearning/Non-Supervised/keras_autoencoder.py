import tensorflow as tf
from keras.api.layers import Input, Dense
from keras.api.models import Model
from keras.api.datasets.mnist import load_data

# Define encoder
input_layer = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_layer)
# Define decoder
decoded = Dense(794, activation='sigmoid')(encoded)
# Combine the encoder and decoder into an autoencoder model
autoencoder = Model(input_layer, decoded)
# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
print(autoencoder.summary())

(x_train, _), (x_test, _) = load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), 784))
x_train = x_train.reshape((len(x_test), 784))

# Train the autoencoder
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
