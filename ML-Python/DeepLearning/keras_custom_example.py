import tensorflow as tf
from keras.api.layers import Layer, Dropout
from keras.api.models import Sequential

class CustomDenseLayer(Layer):
    def __init__(self, units=128):
        super(CustomDenseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True)
        
    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
    
from keras.api.layers import Softmax

model = Sequential([
    CustomDenseLayer(64),
    Dropout(rate=0.5),
    CustomDenseLayer(10),
    Softmax()
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
print("Model summary before building")
print(model.summary())

model.build((1000, 20))
print("\nModel summary after building:")
print(model.summary())

import numpy as np 

# Generate random data 
x_train = np.random.random((1000, 20)) 
y_train = np.random.randint(10, size=(1000, 1)) 

# Convert labels to categorical one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10) 
model.fit(x_train, y_train, epochs=10, batch_size=32)
# Generate random test data 
x_test = np.random.random((200, 20)) 
y_test = np.random.randint(10, size=(200, 1)) 

# Convert labels to categorical one-hot encoding 
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10) 

# Evaluate the model 
loss = model.evaluate(x_test, y_test) 
print(f'Test loss: {loss}')

from keras.api.utils import plot_model
model_plot_file = "model.png"
plot_model(model, to_file=model_plot_file, show_shapes=True)
