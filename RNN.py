import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'      # to silence some warnings

import tensorflow as tf
from tensorflow import keras            # kera is an API 
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()   # it will returns two tuples
x_train, x_test = x_train / 255.0, x_test / 255.0
# 28, 28
# input_size = 28
# seq_lenght = 28  (it means we have 28 time steps in our sequence and each time step we have 28 feature)

# Model 
model = keras.models.Sequential()
model.add(keras.Input(shape=(28, 28)))      # seq_length, input_size
model.add(layers.SimpleRNN(128, return_sequences=False))            # N, 128
model.add(layers.Dense(10))
print(model.summary())
# import sys; sys.exit()

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]  # defice metric that we want to track

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# training
batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)

# evaluate
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
