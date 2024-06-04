'''
    The following script builds, trains, and evaluates a neural network from scratch 
    to classify images in handwritten digits in the MNIST dataset.
    Author: Rohit Rajagopal
'''


import numpy as np
from keras.datasets import mnist
import os


class NeuralNetwork:

    # Ensure layers are added sequentially or the network will need to be rebuilt from scratch
    # Error used is Cross Entropy loss as this is a multi-classification problem with a softmax 
    # output activation function
    # Needs at least 1 hidden layer to work
    # Data must be reshaped such that each row is one example

    def __init__(self):

        # Intiialise all parameters in the neural network
        self.input_size = 0
        self.input_layer = []
        self.hidden_layers = []
        self.activations = []
        self.output_layer = []
        self.weights = []
        self.biases = []
        self.rmsweights = []
        self.rmsbiases = []

    def add_input_layer(self, input_size):
        
        # Initialise number of neurons in input layer
        self.input_size = input_size
    

    def add_hidden_layer(self, hidden_size):

        # Append new arrays to hidden layer, activations and biases
        self.hidden_layers.append(np.zeros(hidden_size))
        self.activations.append(np.zeros(hidden_size))
        self.biases.append(np.zeros(hidden_size))

        # Add weights based on layers already added to neural network (He intialisation for ReLU activation function)
        if len(self.hidden_layers) == 1:
            self.weights.append(np.random.rand(hidden_size, self.input_size) * np.sqrt(2 / self.input_size))
        else:
            self.weights.append(np.random.rand(hidden_size, len(self.hidden_layers[-2])) * np.sqrt(2 / len(self.hidden_layers[-2])))
    

    def add_output_layer(self, output_size):

        # Append new arrays to output layer, activations and biases
        self.output_layer = np.zeros(output_size)
        self.activations.append(np.zeros(output_size))
        self.biases.append(np.zeros(output_size))

        # Add weights for final layer (Xavier initialisation for Softmax activation function)
        self.weights.append(np.random.rand(output_size, len(self.hidden_layers[-1])) * np.sqrt(1 / len(self.hidden_layers[-1])))


    def forward_propagation(self):
        
        # Calculate activated neurons for each of the hidden layer using the ReLU activation function
        for i in range(len(self.hidden_layers)):
            if i == 0:
                self.hidden_layers[i] = np.dot(self.weights[i], self.input_layer)
                self.activations[i] = NeuralNetwork.relu(self.hidden_layers[i])
            else:
                self.hidden_layers[i] = np.dot(self.weights[i], self.activations[i - 1])
                self.activations[i] = NeuralNetwork.relu(self.hidden_layers[i])

        # Calculate the final output using the Softmax activation function
        self.output_layer = np.dot(self.weights[-1], self.activations[-2])
        self.activations[-1] = NeuralNetwork.softmax(self.output_layer)

    
    def backward_propagation(self, label_vector, batch_size, learning_rate, decay_rate):

        # Find the difference between model output and actual labels
        delta_output = self.activations[-1] - label_vector

        # Update weights and biases in each layer, working backwards from the output
        for i in range(len(self.weights) - 1, -1, -1):
            dW_list = []
            dB_list = []
            
            # Consider output layer
            if i == len(self.weights) - 1:

                # Find output weights and biases gradients for each example in the batch and compute the average 
                for j in range(batch_size):
                    dW_list.append(np.outer(delta_output[:, j], self.activations[i - 1][:, j]))
                    dB_list.append(delta_output[:, j])

                delta_hidden = delta_output

            # Consider input layer
            elif i == 0:
                
                # Find input weights and biases gradients for each example in the batch and compute the average 
                delta_hidden = np.dot(self.weights[i + 1].T, delta_hidden) * (self.hidden_layers[i] > 0)
                for j in range(batch_size):
                    dW_list.append(np.outer(delta_hidden[:, j], self.input_layer[:, j]))
                    dB_list.append(delta_hidden[:, j])

            # Consider hidden layers
            else:

                # Find hidden weights and biases gradients for each example in the batch and compute the average 
                delta_hidden = np.dot(self.weights[i + 1].T, delta_hidden) * (self.hidden_layers[i] > 0)
                for j in range(batch_size):
                    dW_list.append(np.outer(delta_hidden[:, j], self.activations[i - 1][:, j]))
                    dB_list.append(delta_hidden[:, j])

            # Update weights and biases
            dW = np.mean(dW_list, axis = 0)
            dB = np.mean(dB_list, axis = 0)
            update_weights, update_biases = NeuralNetwork.rmsprop(self, i, dW, dB, learning_rate, decay_rate)
            self.weights[i] -= update_weights
            self.biases[i] -= update_biases

    
    def rmsprop(self, layer, gradient_weights, gradient_biases, learning_rate, decay_rate):

        epsilon = 1e-8

        # Update exponentially decaying average of squared gradients for weights and biases
        self.rmsweights[layer] = decay_rate * self.rmsweights[layer] + (1 - decay_rate) * gradient_weights ** 2
        self.rmsbiases[layer] = decay_rate * self.rmsbiases[layer] + (1 - decay_rate) * gradient_biases ** 2

        # Update weights and biases using RMSprop update rule
        update_weights = (learning_rate / (np.sqrt(self.rmsweights[layer]) + epsilon)) * gradient_weights
        update_biases = (learning_rate / (np.sqrt(self.rmsbiases[layer]) + epsilon)) * gradient_biases

        return update_weights, update_biases


    def train_network(self, training_data, labels, batch_size, learning_rate, decay_rate, epochs):

        # Initialise RMSProp decaying averages
        self.rmsweights = [np.zeros_like(weights) for weights in self.weights]
        self.rmsbiases = [np.zeros_like(biases) for biases in self.biases]

        # Iterate through each batch in the training set and repeat based on the number of epochs
        for epoch in range(epochs):
            
            print("Commencing Epoch:", epoch + 1)

            # Shuffle training data
            indices = np.arange(training_data.shape[0])
            np.random.shuffle(indices)
            training_data = training_data[indices]
            labels = labels[indices]

            # Iterate through each batch
            for batch in range(0, len(indices), batch_size):
                x_batch = training_data[batch:batch + batch_size]
                y_batch = labels[batch:batch + batch_size]

                # Create one-hot encoded matrix representation of labels
                y_batch_vector = np.eye(self.output_layer.shape[0])[y_batch].T

                # Pass data into neural network
                self.input_layer = x_batch.T
                self.forward_propagation()
                self.backward_propagation(y_batch_vector, batch_size, learning_rate, decay_rate)
        pass


    def predict(self, data):

        pass


    def softmax(inactive):
        
        # Find exponential scores for each element
        exp_scores = np.exp(inactive - np.max(inactive, axis = 0, keepdims = True))

        # Calculate probability distribution
        activated = exp_scores / np.sum(exp_scores, axis = 0, keepdims = True)

        return activated
        

    def relu(inactive): 

        # Find the maximum of 0 and each element
        activated = np.maximum(0, inactive)
        
        return activated




(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1] ** 2))/255
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1] ** 2))/255

new = NeuralNetwork()
new.add_input_layer(784)
new.add_hidden_layer(256)
new.add_hidden_layer(128)
new.add_output_layer(10)
new.train_network(train_X, train_y, 32, 0.01, 0.9, 10)


# add all layers
layer_1_size = 2
layer_1_weights = np.random.rand(layer_1_size, len(test_vars))
layer_2_weights = np.random.rand(n_labels, layer_1_size)
bias_1 = np.zeros(layer_1_size)
bias_2 = np.zeros(n_labels)

# # forward propagation
# layer_1 = np.dot(layer_1_weights, test_vars) + bias_1
# layer_1_neurons = np.maximum(0, layer_1)
# layer_2 = np.dot(layer_2_weights, layer_1_neurons) + bias_2
# output = softmax(layer_2)

# # create one hot encoded comparison set

# # backward propagation (use cross entropy loss)
# learning_rate = 0.05
# delta_output = output - comp
# layer_2_weights -= learning_rate * np.outer(delta_output, layer_1_neurons)
# bias_2 -= learning_rate * delta_output
# delta_hidden = np.dot(layer_2_weights.T, delta_output) * (layer_1_neurons > 0)
# layer_1_weights -= learning_rate * np.outer(delta_hidden, test_vars)
# bias_1 -= learning_rate * delta_hidden




