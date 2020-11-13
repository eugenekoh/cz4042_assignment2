import time

import matplotlib.pyplot as plt

from constants import IMG_PATH, EPOCHS

IMG_PATH.mkdir(parents=True, exist_ok=True)


class History:
    def __init__(self, model_name):
        self.model_name = model_name
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

    def start(self):
        self.start_time = time.time()

    def end(self):
        total_duration = time.time() - self.start_time
        avg_time = total_duration / EPOCHS
        return avg_time


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

    file_path = IMG_PATH / ("loss_" + history.model_name)
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

    file_path = IMG_PATH / ("accuracy_" + history.model_name)
    plt.savefig(file_path)
    plt.close()
