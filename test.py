import tensorflow as tf
import keras
import numpy as np

class CustomModel(keras.Model):
    def train_step(self, data):

        x,y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            loss = self.compute_loss(y=y, y_pred=y_pred)


        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}



# Construct and compile an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
#adama = tf.keras.optimizers.Adam(lr)
#loss = tf.keras.losses.binary_crossentropy
#model.compile(optimizer=adama, loss=loss, metrics=['accuracy'])
#model.summary()
model.compile(optimizer="adam", loss="mse", metrics=["mse"])

# Just use `fit` as usual
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=3)    
        
