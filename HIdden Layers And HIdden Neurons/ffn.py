import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'      # to silence some warnings

import tensorflow as tf
from tensorflow import keras            # kera is an API 
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()   # it will returns two tuples
print(x_train.shape)
print(y_train.shape)        # the values of range between 0 to 255. We will normalize the data so that values becomes between 0 and 1

# normalize 
x_train, x_test = x_train / 255.0, x_test / 255.0

# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(x_train[i], cmap='gray')
# plt.show()

# Model (We will use Sequential API)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'), # relu includes non-linearity and it improves training, so it makes our model better
    keras.layers.Dense(10)   
])

print(model.summary())

# model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=(28, 28)))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(10))

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

# predictions (for predictions we need softmax layer to call probability)
probability_model = keras.models.Sequential([
    model, 
    keras.layers.Softmax()
])

predictions = probability_model(x_test)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

# model + softmax
predictions = model.predict(x_test, batch_size=batch_size)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

pred05s = predictions[0:5]
print(pred05s.shape)
label05s = np.argmax(pred05s, axis=1)
print(label05s)


