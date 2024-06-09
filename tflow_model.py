'''
    The following script uses TensorFlow model architecture to build, train, and evaluate a 
    neural network for classifying handwritten digits in the MNIST dataset.
    Author: Rohit Rajagopal
'''


import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import History


class TflowModel:

    def __init__(self):

        # Initialise weights, biases, and hidden layer sizes
        self.weights_initialiser = tf.initializers.GlorotUniform()
        self.biases_initialiser = "zeros"
        self.hidden_layer_sizes = []

        # Initialise model
        self.model = tf.keras.Sequential()

    def add_input_layer(self, input_size):

        # Initialise number of values in input layer
        self.input_size = input_size

    def add_hidden_layer(self, hidden_size):

        self.hidden_layer_sizes.append(hidden_size)

        # For first hidden layer, consider weights and biases between input and itself
        if len(self.hidden_layer_sizes) == 1:
            self.model.add(tf.keras.layers.Dense(hidden_size, activation = 'relu', kernel_initializer = self.weights_initialiser, 
                                                 bias_initializer = self.biases_initialiser, input_shape = (self.input_size,)))
        
        # For other hidden layers
        else:
            self.model.add(tf.keras.layers.Dense(hidden_size, activation = 'relu', kernel_initializer = self.weights_initialiser, 
                                                 bias_initializer = self.biases_initialiser))
    
    def add_output_layer(self, output_size):

        # Add weights and biases for output layer
        self.model.add(tf.keras.layers.Dense(output_size, activation = 'softmax', kernel_initializer = self.weights_initialiser,
                                             bias_initializer = self.biases_initialiser))

    def train_network(self, x_train, y_train, batch_size, learning_rate, decay_rate, epochs):
        
        # Compile the model
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate, rho = decay_rate)
        self.model.compile(optimizer = optimizer,
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

        # Define a History callback to record training metrics
        history = History()

        # Train the model
        self.model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, callbacks = [history])

        # Print model summary
        self.model.summary()

        # Extract training accuracies and losses for each epoch
        self.training_accuracies = history.history['accuracy']
        self.training_losses = history.history['loss']

    def predict(self, x_test):

        # Predict on specified dataset
        probabilities = self.model.predict(x_test)

        # Extract confidence scores for predicted labels
        confidence_scores = np.max(probabilities, axis = 1)

        # Extract predicted labels
        y_pred = np.argmax(probabilities, axis = 1)

        return confidence_scores, y_pred
    