'''
    The following script contains functions to compute various metrics using model outputs
    and evaluate their performances.
    Author: Rohit Rajagopal
'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def classification_report_ad(true_labels, predicted_labels, save_dir):

    """
        Generate a classification report of key metrics for model evaluation.

        Inputs:
            - true_labels: A 1D array-like of true class labels
            - predicted_labels: A 1D array-like of predicted class labels
            - save_dir: A string to indicate directory to save classification report

        Outputs:
            None
    """

    # Generate classification report for micro and macro-averages
    report = classification_report(true_labels, predicted_labels, digits = 4)

    # Display and save report
    print(report)
    with open(save_dir + '/classification_report.txt', 'w') as f:
        f.write(report)

    return

def confidence_distribution(output_probabilities, save_dir, show):

    """
        Plot the distribution of confidence scores for predicted labels.

        Inputs:
            - output_probabilities: A 1D array-like containing confidence scores for predicted labels
            - save_dir: A string to indicate directory to save plot
            - show: Boolean to indicate whether to display plot

        Outputs:
            None
    """

    # Plot and save histogram
    plt.hist(output_probabilities, bins = 10, color = 'skyblue', edgecolor = 'black')
    plt.xlabel('Confidence Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores for Predicted Labels')
    plt.savefig(save_dir + '/confidence_distribution.png')

    # Display plot
    if show:
        plt.show()

    return

def confusion_matrix_ad(true_labels, predicted_labels, num_classes, save_dir, show):
    
    """
        Generate and plot a confusion matrix.

        Inputs:
            - true_labels: A 1D array-like of true class labels
            - predicted_labels: A 1D array-like of predicted class labels
            - num_classes: The number of classes in the classification task
            - save_dir: A string to indicate directory to save confusion matrix
            - show: Boolean to indicate whether to display confusion matrix

        Outputs:
            None
    """

    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, normalize = 'true')

    # Plot and save confusion matrix
    plt.figure(figsize = (10, 8))
    plt.imshow(conf_matrix * 100, cmap = plt.cm.gist_stern)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(np.arange(num_classes))
    plt.yticks(np.arange(num_classes))
    plt.title('Confusion Matrix')
    plt.colorbar().set_label('Percentage (%)')
    plt.savefig(save_dir + '/confusion_matrix.png')

    # Display confusion matrix
    if show:
        plt.show()

    return

def training_progress(epochs, accuracies, losses, save_dir, show):

    """
        Plot training progress over epochs.

        Inputs:
            - epochs: A list of epoch numbers
            - accuracies: A list of accuracy values corresponding to each epoch
            - losses: A list of loss values corresponding to each epoch
            - save_dir: A string to indicate directory to save plot
            - show: Boolean to indicate whether to display plot

        Outputs:
            None
    """

    fig, ax1 = plt.subplots()

    # Plot accuracies
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color = 'tab:blue')
    ax1.plot(epochs, accuracies, color = 'tab:blue', label = 'Accuracy')
    ax1.tick_params(axis = 'y', labelcolor = 'tab:blue')

    # Creating twin Axes
    ax2 = ax1.twinx()

    # Plot losses
    ax2.set_ylabel('Loss', color = 'tab:red')
    ax2.plot(epochs, losses, color = 'tab:red', label = 'Loss')
    ax2.tick_params(axis = 'y', labelcolor = 'tab:red')

    fig.tight_layout()
    plt.title('Accuracy and Average Loss per Epoch')
    plt.savefig(save_dir + '/training_progress.png')

    # Display plot
    if show:
        plt.show()

    return
