'''
    The following script builds, trains, and evaluates a neural network from scratch 
    to classify images in handwritten digits in the MNIST dataset.
    Author: Rohit Rajagopal
'''


import numpy as np
from keras.datasets import mnist


class NeuralNetwork:

    # Ensure layers are added sequentially or the network will need to be rebuilt from scratch
    # Error used is Cross Entropy loss as this is a multi-classification problem with a softmax 
    # output activation function
    # Needs at least 1 hidden layer to work

    def __init__(self):

        # Intiialise all parameters in the neural network
        self.input_layer = []
        self.hidden_layers = []
        self.activations = []
        self.output_layer = []
        self.weights = []
        self.biases = []


    def add_input_layer(self, input_size):
        
        # Create a new array for the input layer
        self.input_layer = np.zeros(input_size)
    

    def add_hidden_layer(self, hidden_size):

        # Append new arrays to hidden layer, activations and biases
        self.hidden_layers.append(np.zeros(hidden_size))
        self.activations.append(np.zeros(hidden_size))
        self.biases.append(np.zeros(hidden_size))

        # Add weights based on layers already added to neural network
        if len(self.hidden_layers) == 1:
            self.weights.append(np.random.rand(input_size, len(self.input_layer)))
        else:
            self.weights.append(np.random.rand(input_size, len(self.hidden_layers[-1])))
    

    def add_output_layer(self, output_size):

        # Append new arrays to output layer, activations and biases
        self.output_layer = np.zeros(output_size)
        self.activations.append(np.zeros(output_size))
        self.biases.append(np.zeros(output_size))

        # Add weights for final layer
        self.weights.append(np.random.rand(output_size, len(self.hidden_layers[-1])))


    def forward_propagation(self):
        
        # Calculate activated neurons for each of the hidden layer using the ReLU activation function
        for i in range(len(self.hidden_layers)):
            if i == 0:
                self.hidden_layers[i] = np.dot(self.weights[i], self.input_layer)
                self.activations[i] = relu(self.hidden_layers[i])
            else:
                self.hidden_layers[i] = np.dot(self.weights[i], self.activations[i - 1])
                self.activations[i] = relu(self.hidden_layers[i])

        # Calculate the final output using the Softmax activation function
        self.output_layer = np.dot(self.weights[-1], self.activations[-1])
        self.activations[-1] = softmax(self.output_layer)

    
    def backward_propagation(self, label_vector, optimiser, optimiser_params):

        # Find the difference between model output and actual labels
        delta_output = self.activations[-1] - label_vector

        # Find gradients for each of the weights and biases 
        dW_output = np.outer(delta_output, self.activations[-2])
        dB_output = delta_output

        for i in range(len(self.hidden_layers)):
            self.weights[-1] = 
            self.biases[-1]



        # Update all layer weights and biases
        if optimiser == 'RMSProp':
            self.weights[-1], self.biases[-1] = rmsprop(self, delta_output, optimiser_params)
        elif optimiser == 'Gradient Descent':
            self.weights[-1], self.biases[-1] = gradient_descent(self, optimiser_params)            

        learning_rate = 0.05
        delta_output = output - comp
        layer_2_weights -= learning_rate * np.outer(delta_output, layer_1_neurons)
        bias_2 -= learning_rate * delta_output
        delta_hidden = np.dot(layer_2_weights.T, delta_output) * (layer_1_neurons > 0)
        layer_1_weights -= learning_rate * np.outer(delta_hidden, test_vars)
        bias_1 -= learning_rate * delta_hidden

    
    def rmsprop(self, optimiser_params):


        pass

    def gradient_descent(self, learning_rate):

        pass


    def train_network(self, training_data, labels, batch_size, optimiser, optimiser_params, epochs):

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


    def one_hot_encode(label, output_size):

        encoded_vector = np.zeros(output_size)
        encoded_vector[label - 1] = 1

        return encoded_vector






    def forward(self, x):
        activations = [x]
        for i in range(len(self.weights)):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)  # ReLU activation for hidden layers
            else:
                x = self.softmax(x)    # Softmax activation for output layer
            activations.append(x)
        return activations

    def backward(self, x, y, learning_rate=0.01):
        activations = self.forward(x)
        output = activations[-1]

        # Compute gradients for output layer
        delta = output - y
        gradients = [np.dot(activations[-2].T, delta)]

        # Backpropagate through hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * (activations[i+1] > 0)  # ReLU derivative
            gradients.insert(0, np.dot(activations[i].T, delta))

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i]
            self.biases[i] -= learning_rate * np.sum(gradients[i], axis=0)

    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Example usage:
# Define the neural network architecture
input_size = 784  # MNIST images are 28x28 pixels
hidden_sizes = [128, 64]
output_size = 10  # 10 classes (digits 0-9)

# Create the neural network
model = NeuralNetwork(input_size, hidden_sizes, output_size)

# Train the neural network
# Assuming x_train contains input images and y_train contains corresponding labels
# Perform forward pass, compute loss, perform backward pass, and update parameters iteratively
for epoch in range(num_epochs):
    for x, y in zip(x_train, y_train):
        # Forward pass
        activations = model.forward(x)
        
        # Compute loss
        loss = compute_loss(activations[-1], y)
        
        # Backward pass
        model.backward(x, y, learning_rate)






def softmax(z):

    exp_scores = np.exp(z - np.max(z, axis=0, keepdims=True))
    softmax = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    return softmax

def one_hot_encode(index, length):
    encoded_vector = np.zeros(length)
    encoded_vector[index - 1] = 1
    return encoded_vector

(train_X, train_y), (test_X, test_y) = mnist.load_data()
test_vars = np.reshape(train_X[0], 784)/255
test_vars = test_vars[176:179]
label = 2 #train_y[0]
n_labels = 2

# add all layers
layer_1_size = 2
layer_1_weights = np.random.rand(layer_1_size, len(test_vars))
layer_2_weights = np.random.rand(n_labels, layer_1_size)
bias_1 = np.zeros(layer_1_size)
bias_2 = np.zeros(n_labels)

# forward propagation
layer_1 = np.dot(layer_1_weights, test_vars) + bias_1
layer_1_neurons = np.maximum(0, layer_1)
layer_2 = np.dot(layer_2_weights, layer_1_neurons) + bias_2
output = softmax(layer_2)

# create one hot encoded comparison set
comp = one_hot_encode(label, n_labels)

# backward propagation (use cross entropy loss)
learning_rate = 0.05
delta_output = output - comp
layer_2_weights -= learning_rate * np.outer(delta_output, layer_1_neurons)
bias_2 -= learning_rate * delta_output
delta_hidden = np.dot(layer_2_weights.T, delta_output) * (layer_1_neurons > 0)
layer_1_weights -= learning_rate * np.outer(delta_hidden, test_vars)
bias_1 -= learning_rate * delta_hidden




