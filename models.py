import tensorflow as tf
from tensorflow.keras import Model, layers

NUM_CLASSES = 15
DROP_RATE = 0.4


class CharCNN(Model):
    def __init__(self, vocab_size=256, drop_out=False):
        super(CharCNN, self).__init__()
        self.vocab_size = vocab_size
        self.drop_out = drop_out

        # Weight variables and cells
        self.conv1 = layers.Conv2D(filters=10, kernel_size=(20, 256), padding='valid', activation='relu', use_bias=True,
                                   input_shape=(100, 256, 1))
        self.pool1 = layers.MaxPool2D(pool_size=4, strides=2, padding='same')
        self.conv2 = layers.Conv2D(filters=10, kernel_size=(20, 1), padding='valid', activation='relu', use_bias=True)
        self.pool2 = layers.MaxPool2D(pool_size=4, strides=2, padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(units=NUM_CLASSES, activation='softmax')

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

    def model_name(self):
        return f"char_cnn{'_dropout' if self.drop_out else ''}"


class CharRNN(Model):
    def __init__(self, vocab_size=256, drop_out=False, two_layer=False, rnn_layer=layers.GRU):
        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.drop_out = drop_out
        self.two_layer = two_layer

        self.rnn1 = rnn_layer(units=20)
        self.rnn2 = rnn_layer(units=20)
        self.dense = layers.Dense(units=15, activation='softmax')

    def call(self, x, drop_rate=0.4):
        # forward logic
        # forward
        x = tf.one_hot(x, self.vocab_size)
        x = self.rnn1(x)
        if self.two_layer:
            x = self.rnn2(x)
        if self.drop_out:
            x = layers.Dropout(drop_rate)(x)
        logits = self.dense(x)

        return logits

    def model_name(self):
        return f"char_rnn{'_dropout' if self.drop_out else ''}"


class WordRNN(Model):

    def __init__(self, vocab_size, hidden_dim=10):
        super(WordRNN, self).__init__()
        # Hyperparameters
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = layers.Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_DOCUMENT_LENGTH)
        # Weight variables and RNN cell
        self.rnn = layers.RNN(
            tf.keras.layers.GRUCell(self.hidden_dim), unroll=True)
        self.dense = layers.Dense(MAX_LABEL, activation=None)

    def call(self, x, drop_rate):
        # forward logic
        embedding = self.embedding(x)
        encoding = self.rnn(embedding)

        encoding = tf.nn.dropout(encoding, drop_rate)
        logits = self.dense(encoding)

        return logits
