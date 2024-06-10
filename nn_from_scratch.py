"""
    The following script contains the NeuralNetwork class to build, train, and evaluate a 
    neural network from scratch.
    Author: Rohit Rajagopal
"""


import numpy as np
from tqdm import tqdm


class NeuralNetwork:

    """
        A class for building, training, and evaluating a neural network based on user 
        inputs.

        Attributes:
            - hidden_layer_sizes (list): Sizes of the hidden layers.
            - weights (list): Weights of the neural network.
            - biases (list): Biases of the neural network.
            - activations (list): Activations of the neural network layers.
            - training_accuracies (list): Training accuracies per epoch.
            - training_losses (list): Training losses per epoch.
            - epsilon (float): Small value to avoid mathematical inoperations.
            
        Notes:
            - Ensure layers are added sequentially or the network will need to be rebuilt from scratch.
            - Loss is calculated using the Cross-Entropy function as this network is standardized for
              a multi-classification problem with a Softmax activation function in the output layer.
            - ReLU activation function used in hidden layers.
            - Requires at least one hidden layer to function correctly. 
            - Input data must be reshaped such that each row is one training example and the labels 
              are one-hot encoded.
    """

    def __init__(self):
        
        """
            Initializes the neural network with default parameters.
            
            Args:
                None
                
            Returns:
                None
        """

        # Intiialise parameters in the neural network
        self.hidden_layer_sizes = []
        self.weights = []
        self.biases = []
        self.activations = []
        self.training_accuracies = []
        self.training_losses = []
        self.epsilon = 1e-12

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

        # For first hidden layer, consider weights and biases between input layer and itself (Glorot initialisation)
        if len(self.hidden_layer_sizes) == 1:
            bound = np.sqrt(6 / (self.input_size + hidden_size))
            self.weights.append(np.random.uniform(-bound, bound, size = (self.input_size, hidden_size)))
            self.biases.append(np.zeros([1, hidden_size]))

        # For other hidden layers (Glorot initialisation)
        else:
            bound = np.sqrt(6 / (self.hidden_layer_sizes[-2] + hidden_size))
            self.weights.append(np.random.uniform(-bound, bound, size = (self.hidden_layer_sizes[-2], hidden_size)))
            self.biases.append(np.zeros([1, hidden_size]))

    def add_output_layer(self, output_size):
        
        """
            Adds an output layer to the neural network.

            Args:
                - output_size (int): Number of neurons in the output layer.

            Returns:
                None
        """
        
        # Add weights and biases for output layer (Glorot initialisation)
        bound = np.sqrt(6 / (self.hidden_layer_sizes[-1] + output_size))
        self.weights.append(np.random.uniform(-bound, bound, size = (self.hidden_layer_sizes[-1], output_size)))
        self.biases.append(np.zeros([1, output_size]))

    def forward_propagation(self):
        
        """
            Performs forward propagation through the neural network.

            Args:
                None

            Returns:
                None
        """
        
        # Calculate activated neurons for each of the hidden layers using the ReLU activation function
        for layer in range(len(self.hidden_layer_sizes)):
            if layer == 0:
                self.hidden_layers = [np.dot(self.input_layer, self.weights[layer]) + self.biases[layer]]
            else:
                self.hidden_layers.append(np.dot(self.activations[-1], self.weights[layer]) + self.biases[layer])
            
            self.activations.append(NeuralNetwork.relu(self.hidden_layers[layer]))

        # Calculate the final output using the Softmax activation function
        self.output_layer = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.activations.append(NeuralNetwork.softmax(self.output_layer))

    def backward_propagation(self, label_vector, learning_rate, decay_rate):
        
        """
            Performs backward propagation and updates the neural network weights and biases.

            Args:
                - label_vector (array-like): One-hot encoded true class labels.
                - learning_rate (float): Learning rate for weight updates.
                - decay_rate (float): Decay rate for RMSprop.

            Returns:
                None
        """
        
        # Find the difference between model output and actual labels
        delta_output = self.activations[-1] - label_vector

        # Update weights and biases in each layer, working backwards from the output
        for i in range(len(self.hidden_layer_sizes), -1, -1):
            dW = np.zeros_like(self.weights[i])
                                               
            # Consider output layer
            if i == len(self.hidden_layer_sizes):
                dW = np.dot(self.activations[i - 1].T, delta_output)
                dB = np.sum(delta_output, axis = 0, keepdims = True)
                delta_hidden = np.dot(delta_output, self.weights[i].T) * (self.hidden_layers[i - 1] > 0)    # ReLU Derivative

            # Consider input layer
            elif i == 0:
                dW = np.dot(self.input_layer.T, delta_hidden)
                dB = np.sum(delta_hidden, axis = 0, keepdims = True)

            # Consider hidden layers
            else:
                dW = np.dot(self.activations[i - 1].T, delta_hidden)
                dB = np.sum(delta_hidden, axis = 0, keepdims = True)
                delta_hidden = np.dot(delta_hidden, self.weights[i].T) * (self.hidden_layers[i - 1] > 0)    # ReLU Derivative

            # Update weights and biases
            update_weights, update_biases = NeuralNetwork.rmsprop(self, i, dW, dB, learning_rate, decay_rate)
            self.weights[i] -= update_weights
            self.biases[i] -= update_biases

    def rmsprop(self, layer, gradient_weights, gradient_biases, learning_rate, decay_rate):

        """
            Applies RMSprop optimization to update the neural network parameters.

            Args:
                - layer (int): Layer index to update.
                - gradient_weights (array-like): Gradient of the weights.
                - gradient_biases (array-like): Gradient of the biases.
                - learning_rate (float): Learning rate for weight updates.
                - decay_rate (float): Decay rate for RMSprop.

            Returns:
                - update_weights (array-like): Amount to reduce weights for the specified layer.
                - update_biases (array-like): Amount to reduce biases for the specified layer.
        """
        
        # Update exponentially decaying average of squared gradients for weights and biases
        self.rmsweights[layer] = decay_rate * self.rmsweights[layer] + (1 - decay_rate) * gradient_weights ** 2
        self.rmsbiases[layer] = decay_rate * self.rmsbiases[layer] + (1 - decay_rate) * gradient_biases ** 2

        # Update weights and biases using RMSprop update rule
        update_weights = learning_rate * (gradient_weights / (np.sqrt(self.rmsweights[layer]) + self.epsilon))
        update_biases = learning_rate * (gradient_biases / (np.sqrt(self.rmsbiases[layer]) + self.epsilon))

        return update_weights, update_biases

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
        
        # Initialise RMSProp decaying averages
        self.rmsweights = [np.zeros_like(weight) for weight in self.weights]
        self.rmsbiases = [np.zeros_like(bias) for bias in self.biases]

        # Iterate through each batch in the training set and repeat based on the number of epochs
        for epoch in range(epochs):
            self.training_accuracies.append(0)
            self.training_losses.append(0)
            print("\nCommencing Epoch:", str(epoch + 1) + "/" + str(epochs))

            # Shuffle training data
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            # Iterate through each batch
            for batch in tqdm(range(0, len(indices), batch_size), position = 0, leave = True):
                x_batch = x_train[batch:batch + batch_size]
                y_batch = y_train[batch:batch + batch_size]

                # Pass data into neural network
                self.input_layer = x_batch
                self.forward_propagation()
                self.backward_propagation(y_batch, learning_rate, decay_rate)

                # Find accuracy (ratio of predicted correct to number of examples)
                y_pred = np.argmax(self.activations[-1], axis = 1)
                y_labels = np.argmax(y_batch, axis = 1)
                self.training_accuracies[epoch] += np.count_nonzero((y_pred - y_labels) == 0)

                # Find cross entropy loss
                y_pred_conf = np.clip(self.activations[-1], self.epsilon, 1)    # Ensure log(0) doesn't result in an error
                self.training_losses[epoch] += np.sum(-np.sum(y_batch * np.log(y_pred_conf), axis = 1))

            # Print accuracy and loss metrics at the end of each epoch
            self.training_accuracies[epoch] = self.training_accuracies[epoch] / len(indices)
            print("Accuracy: %.4f" % self.training_accuracies[epoch])
            
            self.training_losses[epoch] = self.training_losses[epoch] / len(indices)
            print("Average Cross-Entropy Loss: %.4f" % self.training_losses[epoch])

    def predict(self, data):
        
        """
            Predicts labels for the given data.

            Args:
                - data (array-like): Data to predict labels for.

            Returns:
                - confidence_scores (array-like): Confidence scores/probabilties for each predicted class.
                - y_pred (array-like): Predicted labels.
        """
        
        # Apply forward propagation using updated model weights
        self.input_layer = data
        self.forward_propagation()

        # Extract confidence scores of predicted labels
        confidence_scores = np.max(self.activations[-1], axis = 1)

        # Extract the predicted labels
        y_pred = np.argmax(self.activations[-1], axis = 1)

        return confidence_scores, y_pred

    @staticmethod
    def softmax(inactive):
        
        """
        Applies the softmax activation function to the input data.

        Args:
            - inactive (array-like): Input data.

        Returns:
            - activated (array-like): Softmax activated data.
        """
        
        # Subtract the maximum value for numerical stability
        inactive -= np.max(inactive, axis = 1, keepdims = True)
        
        # Find exponential scores for each element
        exp_scores = np.exp(inactive)

        # Calculate probability distribution
        activated = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)

        return activated
        
    @staticmethod
    def relu(inactive): 
        
        """
        Applies the ReLU activation function to the input data.

        Args:
            - inactive (array-like): Input data.

        Returns:
            - activated(array-like): ReLU activated data.
        """
        
        # Find the maximum of 0 and each element
        activated = np.maximum(0, inactive)
        
        return activated
