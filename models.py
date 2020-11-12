import tensorflow as tf
from tensorflow.keras import Model, layers


class CharCNN(Model):
    def __init__(self, vocab_size=256, drop_out=False):
        super(CharCNN, self).__init__()
        self.vocab_size = vocab_size
        self.drop_out = drop_out
        self.model_name = f"char_cnn{'_dropout' if self.drop_out else ''}"

        # Weight variables and cells
        self.conv1 = layers.Conv2D(filters=10, kernel_size=(20, 256), padding='VALID', activation='relu', use_bias=True,
                                   input_shape=(100, 256, 1))
        self.pool1 = layers.MaxPool2D(pool_size=4, strides=2, padding='SAME')
        self.conv2 = layers.Conv2D(filters=10, kernel_size=(20, 1), padding='VALID', activation='relu', use_bias=True)
        self.pool2 = layers.MaxPool2D(pool_size=4, strides=2, padding='SAME')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(units=15, activation='softmax')

    def call(self, x, drop_rate=0.4):
        # forward
        x = tf.one_hot(x, self.vocab_size)
        x = x[..., tf.newaxis]
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        if self.drop_out:
            x = layers.Dropout(drop_rate)(x)
        logits = self.dense(x)

        return logits


class CharRNN(Model):
    def __init__(self, vocab_size=256, drop_out=False, rnn_layer=layers.GRU):
        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.drop_out = drop_out
        self.model_name = f"char_{rnn_layer(1).name}{'_dropout' if self.drop_out else ''}"

        self.rnn1 = rnn_layer(units=20)
        self.dense = layers.Dense(units=15, activation='softmax')

    def call(self, x, drop_rate=0.4):
        # forward
        x = tf.one_hot(x, self.vocab_size)
        x = self.rnn1(x)
        if self.drop_out:
            x = layers.Dropout(drop_rate)(x)
        logits = self.dense(x)

        return logits


class Char2RNN(Model):
    def __init__(self, vocab_size=256, rnn_layer=layers.GRU):
        super(Char2RNN, self).__init__()
        self.vocab_size = vocab_size
        self.model_name = f"char_2GRU"

        self.rnn1 = rnn_layer(units=20, return_sequences=True)
        self.rnn2 = rnn_layer(units=20)
        self.dense = layers.Dense(units=15, activation='softmax')

    def call(self, x, drop_rate=0.4):
        # forward
        x = tf.one_hot(x, self.vocab_size)
        x = self.rnn1(x)
        x = self.rnn2(x)
        logits = self.dense(x)

        return logits


class WordCNN(Model):
    def __init__(self, vocab_size, drop_out=False):
        super(WordCNN, self).__init__()
        self.drop_out = drop_out
        self.model_name = f"word_cnn{'_dropout' if self.drop_out else ''}"

        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=20, input_length=100)
        self.conv1 = layers.Conv2D(filters=10, kernel_size=(20, 20), padding='VALID', activation='relu', use_bias=True)
        self.pool1 = layers.MaxPool2D(pool_size=4, strides=2, padding='SAME')
        self.conv2 = layers.Conv2D(filters=10, kernel_size=(20, 1), padding='VALID', activation='relu', use_bias=True)
        self.pool2 = layers.MaxPool2D(pool_size=4, strides=2, padding='SAME')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(15, activation='softmax')

    def call(self, x, drop_rate=0.5):
        # forward
        # x = tf.one_hot(x, one_hot_size)
        x = self.embedding(x)
        x = x[..., tf.newaxis]
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        if self.drop_out:
            x = layers.Dropout(drop_rate)(x)
        logits = self.dense(x)
        return logits


class WordRNN(Model):
    def __init__(self, vocab_size, drop_out=False, rnn_layer=layers.GRU):
        super(WordRNN, self).__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=20, input_length=100)
        self.drop_out = drop_out
        self.model_name = f"word_{rnn_layer(1).name}{'_dropout' if self.drop_out else ''}"

        self.rnn1 = rnn_layer(units=20)
        self.dense = layers.Dense(units=15, activation='softmax')

    def call(self, x, drop_rate=0.4):
        # forward
        x = self.embedding(x)
        x = self.rnn1(x)
        if self.drop_out:
            x = layers.Dropout(drop_rate)(x)
        logits = self.dense(x)

        return logits


class Word2RNN(Model):
    def __init__(self, vocab_size, rnn_layer=layers.GRU):
        super(Word2RNN, self).__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=20, input_length=100)
        self.model_name = f"word_2GRU"

        self.rnn1 = rnn_layer(units=20, return_sequences=True)
        self.rnn2 = rnn_layer(units=20)
        self.dense = layers.Dense(units=15, activation='softmax')

    def call(self, x, drop_rate=0.4):
        # forward
        x = self.embedding(x)
        x = self.rnn1(x)
        x = self.rnn2(x)
        logits = self.dense(x)

        return logits
