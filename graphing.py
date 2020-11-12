from pathlib import Path

import matplotlib.pyplot as plt

IMG_PATH = Path('results/graphs')


class History:
    def __init__(self, model_name):
        self.model_name = model_name
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []


def plot_loss(history):
    train_loss = history.train_loss
    val_loss = history.test_loss

    # prep figure
    plt.grid()

    # add data points
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Test')

    # labels
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    file_path = IMG_PATH / "partA"/ history.model_name
    plt.savefig(file_path)


# plot the training accuracies and test accuracies against the learning epochs
def plot_accuracies(history):
    train_acc = history.train_acc
    test_acc = history.test_acc

    # prep figure
    plt.grid()

    # add data points
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')

    # labels
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    file_path = IMG_PATH / "partA" / history.model_name
    plt.savefig(file_path)


def plot_train_loss_test_acc(history):
    train_loss = history.train_loss
    test_acc = history.test_acc

    # prep figure
    plt.grid()

    # add data points
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_acc, label='Test Accuracy')

    # labels
    plt.title('Training Loss and Testing Accuracy against Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    file_path = IMG_PATH / "partB" / history.model_name
    plt.savefig(file_path)
