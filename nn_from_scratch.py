'''
    The following script builds, trains, and evaluates a neural network from scratch 
    to classify images in handwritten digits in the MNIST dataset.
    Author: Rohit Rajagopal
'''


import numpy as np
from keras.datasets import mnist

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




