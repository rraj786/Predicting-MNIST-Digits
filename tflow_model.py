'''
    The following script uses TensorFlow model architecture to build, train, and evaluate a 
    neural network for classifying handwritten digits in the MNIST dataset.
    Author: Rohit Rajagopal
'''


import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import History


class TflowModel:
    
    """
        A TensorFlow model class for building, training, and evaluating a neural network
        to classify handwritten digits in the MNIST dataset.

        Attributes:
            - weights_initialiser: Initialiser for the weights using the Glorot method.
            - biases_initialiser: Initialiser for the biases.
            - hidden_layer_sizes (list): Sizes of the hidden layers.
            - model: TensorFlow Sequential model.
    """

    def __init__(self):
        
        """
            Initializes the TensorFlow model with default parameters.

            Args:
                None

            Returns:
                None
        """
    
        # Initialise weights, biases, and hidden layer sizes
        self.weights_initialiser = tf.initializers.GlorotUniform()
        self.biases_initialiser = "zeros"
        self.hidden_layer_sizes = []

        # Initialise model
        self.model = tf.keras.Sequential()

    def add_input_layer(self, input_size):
        
        """
            Adds an input layer to the neural network.

            Args:
                - input_size (int): Number of neurons in the input layer.

            Returns:
                None
        """
        
        # Initialise number of values in input layer
        self.input_size = input_size

    def add_hidden_layer(self, hidden_size):

        """
            Adds a hidden layer to the neural network.

            Args:
                - hidden_size (int): Number of neurons in the hidden layer.

            Returns:
                None
        """
        
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

        """
            Adds an output layer to the neural network.

            Args:
                - output_size (int): Number of neurons in the output layer.

            Returns:
                None
        """
        
        # Add weights and biases for output layer
        self.model.add(tf.keras.layers.Dense(output_size, activation = 'softmax', kernel_initializer = self.weights_initialiser,
                                             bias_initializer = self.biases_initialiser))

    def train_network(self, x_train, y_train, batch_size, learning_rate, decay_rate, epochs):
        
        """
            Trains the neural network using the given training data.

            Args:
                - x_train (array-like): Training data features.
                - y_train (array-like): Training data labels.
                - batch_size (int): Size of the training batches.
                - learning_rate (float): Learning rate for weight updates.
                - decay_rate (float): Decay rate for RMSprop.
                - epochs (int): Number of training epochs.

            Returns:
                None
        """
        
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

    def predict(self, data):

        """
            Predicts labels for the given data.

            Args:
                - data (array-like): Data to predict labels for.

            Returns:
                - confidence_scores (array-like): Confidence scores/probabilties for each predicted class.
                - y_pred (array-like): Predicted labels.
        """
        
        # Predict on specified dataset
        probabilities = self.model.predict(data)

        # Extract confidence scores for predicted labels
        confidence_scores = np.max(probabilities, axis = 1)

        # Extract predicted labels
        y_pred = np.argmax(probabilities, axis = 1)

        return confidence_scores, y_pred
    
