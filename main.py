'''
    The following script compares the results of a neural network made from scratch
    and a TensorFlow model on the MNIST dataset. Note that both share the same model
    architecture.
    Author: Rohit Rajagopal
'''

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os
from nn_from_scratch import NeuralNetwork
from tflow_model import TflowModel
from metrics import *


# Load the MNIST dataset
(x_train, y_train_lab), (x_test, y_test_lab) = mnist.load_data()
print("MNIST Dataset Dimensions")
print("x_train:", x_train.shape)
print("y_train:", y_train_lab.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test_lab.shape)

# Visualise the first 10 images in the training set
plt.figure(figsize = (10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)  
    plt.imshow(x_train[i])
    plt.title(f"Label: {y_train_lab[i]}")
    plt.axis('off')  
plt.tight_layout()
plt.suptitle("Visualising MNIST Digits Dataset (10 Examples)", fontsize = 16)
plt.show()

# Preprocess the data
num_classes = np.max(y_train_lab) + 1
x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.0
x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255.0
y_train = np.eye(num_classes)[y_train_lab]
y_test = np.eye(num_classes)[y_test_lab]

# Set up model parameters
input_neurons = 784
hidden_layer1 = 256
hidden_layer2 = 128
batch_size = 32
learning_rate = 0.001
decay_rate = 0.9
epochs = 10

# Initialise and train model using neural network made from scratch
model_scratch = NeuralNetwork()
model_scratch.add_input_layer(input_neurons)
model_scratch.add_hidden_layer(hidden_layer1)
model_scratch.add_hidden_layer(hidden_layer2)
model_scratch.add_output_layer(num_classes)
model_scratch.train_network(x_train, y_train, batch_size, learning_rate, decay_rate, epochs)

# Initialise and train model using TensorFlow model
model_tflow = TflowModel()
model_tflow.add_input_layer(input_neurons)
model_tflow.add_hidden_layer(hidden_layer1)
model_tflow.add_hidden_layer(hidden_layer2)
model_tflow.add_output_layer(num_classes)
model_tflow.train_network(x_train, y_train, batch_size, learning_rate, decay_rate, epochs)

# Get predictions on test set using both models
y_pred_scratch_scores, y_pred_scratch = model_scratch.predict(x_test)
y_pred_tflow_scores, y_pred_tflow = model_tflow.predict(x_test)

# Set up directories to save metrics
curr_dir = os.getcwd()
scratch_dir = os.path.join(curr_dir, 'metrics/scratch')
tflow_dir = os.path.join(curr_dir, 'metrics/tflow')

# Create new directories if they don't exist already
if not os.path.exists(scratch_dir):
    os.makedirs(scratch_dir)

if not os.path.exists(tflow_dir):
    os.makedirs(tflow_dir)

# Display and compare key metrics for both models based on predictions
epoch_list = list(range(1, epochs + 1))

training_progress(epoch_list, model_scratch.training_accuracies, model_scratch.training_losses, scratch_dir, True)
training_progress(epoch_list, model_tflow.training_accuracies, model_tflow.training_losses, tflow_dir, True)

confusion_matrix_ad(y_test_lab, y_pred_scratch, num_classes, scratch_dir, True)
confusion_matrix_ad(y_test_lab, y_pred_tflow, num_classes, tflow_dir, True)

confidence_distribution(y_pred_scratch_scores, scratch_dir, True)
confidence_distribution(y_pred_tflow_scores, tflow_dir, True)

classification_report_ad(y_test_lab, y_pred_scratch, scratch_dir)
classification_report_ad(y_test_lab, y_pred_tflow, tflow_dir)
