from keras.api.callbacks import Callback

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f'End of epoch {epoch}, loss: {logs.get("loss")}, accuracy: 
              {logs.get("accuracy")}')
        
# Usage in model training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_dataset, epochs=10, callbacks=[CustomCallback()])