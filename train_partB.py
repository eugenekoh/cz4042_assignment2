import csv

import tensorflow as tf
from loguru import logger
from tensorflow.keras import layers

from constants import *
from data_processing import read_data_chars, read_data_words
from graphing import History, plot_loss, plot_accuracies
from models import CharCNN, CharRNN, WordRNN, WordCNN, Char2RNN, Word2RNN

# Module settings
tf.random.set_seed(SEED)
tf.keras.backend.set_floatx('float32')
logger.add("logs/file_{time}.log")


# Build model
def train(model, train_ds, test_ds, clip_value=None):
    # Choose optimizer and loss function for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    if clip_value is not None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=clip_value)

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

    history = History(model.model_name)
    history.start()
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

    avg_epoch_time = history.end()
    # save results
    with open('results/partB_test_accuracies.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([history.model_name, float(max(history.test_acc)), avg_epoch_time])

    # plot graphs
    plot_loss(history)
    plot_accuracies(history)

    return history


if __name__ == '__main__':
    # character models
    logger.debug('retrieving character data')
    train_ds, test_ds, _ = read_data_chars()
    models = [
        CharCNN(),
        CharCNN(drop_out=True),
        CharRNN(),
        CharRNN(drop_out=True),
        CharRNN(rnn_layer=layers.SimpleRNN),
        CharRNN(rnn_layer=layers.LSTM),
        Char2RNN(),
    ]
    for model in models:
        logger.debug(f'training {model.model_name}')
        train(model, train_ds, test_ds)

    # gradient clipping
    model = CharRNN()
    model.model_name += "_clip"
    logger.debug(f'training {model.model_name}')
    train(model, train_ds, test_ds, clip_value=CLIP_VALUE)

    # word models
    logger.debug('retrieving word data')
    train_ds, test_ds, vocab_size = read_data_words()
    models = [
        WordCNN(vocab_size),
        WordCNN(vocab_size, drop_out=True),
        WordRNN(vocab_size),
        WordRNN(vocab_size, drop_out=True),
        WordRNN(vocab_size, rnn_layer=layers.SimpleRNN),
        WordRNN(vocab_size, rnn_layer=layers.LSTM),
        Word2RNN(vocab_size),
    ]
    for model in models:
        logger.debug(f'training {model.model_name}')
        train(model, train_ds, test_ds)

    # gradient clipping
    model = WordRNN(vocab_size)
    model.model_name += "_clip"
    logger.debug(f'training {model.model_name}')
    train(model, train_ds, test_ds, clip_value=CLIP_VALUE)
