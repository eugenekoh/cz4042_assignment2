from loguru import logger
import tensorflow as tf
import csv

from data_processing import read_data_chars
from models import CharCNN, CharRNN
from graphing import plot_train_loss_test_acc, History

ONE_HOT_SIZE = 256
EPOCHS = 2
LR = 0.01
DROP_RATE = 0.4
SEED = 10
tf.random.set_seed(SEED)
tf.keras.backend.set_floatx('float32')

logger.add("logs/file_{time}.log")


# Build model
def train(model, train_ds, test_ds):
    # Choose optimizer and loss function for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=LR)

    # Select metrics to measure the loss and the accuracy of the model.
    # These metrics accumulate the values over epochs and then print the overall result.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Training function
    def train_step(model, x, label, drop_rate):
        with tf.GradientTape() as tape:
            out = model(x, drop_rate)
            loss = loss_object(label, out)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, out)

    # Testing function
    def test_step(model, x, label, drop_rate=0):
        out = model(x, drop_rate)
        t_loss = loss_object(label, out)
        test_loss(t_loss)
        test_accuracy(label, out)

    history = History(model.model_name())
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(model, images, labels, drop_rate=DROP_RATE)

        for images, labels in test_ds:
            test_step(model, images, labels, drop_rate=0)

        history.train_loss.append(train_loss.result())
        history.train_acc.append(train_accuracy.result())
        history.test_loss.append(test_loss.result())
        history.test_acc.append(test_accuracy.result())

        epoch_result = f'Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}, Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result()}'
        logger.debug(epoch_result)

    # save results
    with open('results/test_accuracies.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([history.model_name, max(history.test_acc)])

    return history


if __name__ == '__main__':
    logger.debug('retrieving character data')
    train_ds, test_ds = read_data_chars()

    models = [CharCNN(), CharCNN(drop_out=True), CharRNN(), CharRNN(drop_out=True)]

    for model in models:
        logger.debug(f'training {model.model_name()}')
        history = train(model, train_ds, test_ds)
        plot_train_loss_test_acc(history)