'''
    The following script uses TensorFlow model architecture to build, train, and evaluate a 
    neural network for classifying handwritten digits in the MNIST dataset.
    Author: Rohit Rajagopal
'''


import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.0
x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the neural network architecture
input_size = 784
hidden_size1 = 256
hidden_size2 = 128
output_size = 10

# Initialize weights and biases
initializer = tf.initializers.GlorotUniform()

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size1, activation='relu', kernel_initializer=initializer, input_shape=(input_size,)),
    tf.keras.layers.Dense(hidden_size2, activation='relu', kernel_initializer=initializer),
    tf.keras.layers.Dense(output_size, activation='softmax', kernel_initializer=initializer)
])

# Compile the model
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy}')
