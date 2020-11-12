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

    plt.figure()
    plt.grid()

    # add data points
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Test')

    # labels
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    file_path = IMG_PATH / (history.model_name + "_loss")
    plt.savefig(file_path)
    plt.close()


# plot the training accuracies and test accuracies against the learning epochs
def plot_accuracies(history):
    train_acc = history.train_acc
    test_acc = history.test_acc

    plt.figure()
    plt.grid()

    # add data points
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')

    # labels
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    file_path = IMG_PATH / (history.model_name + "_accuracy")
    plt.savefig(file_path)
    plt.close()


