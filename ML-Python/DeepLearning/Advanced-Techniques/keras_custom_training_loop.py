import keras
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Dense

# Create simple model
model = Sequential([Dense(64, activation='relu'), Dense(10)])
# Custom training loop
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Load or create the training dataset
x_train = tf.random.uniform((100, 10))
y_train = tf.random.uniform((100,), maxval=10, dtype=tf.int64)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

for epoch in range(10):
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')