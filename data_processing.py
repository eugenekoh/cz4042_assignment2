import numpy as np
import collections
import tensorflow as tf
from loguru import logger
from nltk.tokenize import word_tokenize
import csv
import re

BATCH_SIZE = 128
MAX_DOCUMENT_LENGTH = 100

TRAIN_FILE = './data/train_medium.csv'
TEST_FILE = './data/test_medium.csv'


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\" ]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text


def build_char_dict(strings):
    chars = sorted(set(''.join(strings)))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    return vocab_size, char_to_ix


def build_word_dict(contents):
    words = list()
    for content in contents:
        for word in word_tokenize(clean_str(content)):
            words.append(word)

    word_counter = collections.Counter(words).most_common()
    word_dict = dict()
    word_dict["<pad>"] = 0
    word_dict["<unk>"] = 1
    word_dict["<eos>"] = 2
    for word, _ in word_counter:
        word_dict[word] = len(word_dict)
    return word_dict


def preprocess_char(strings, char_to_ix, max_length):
    data_chars = [list(text) for text in strings]
    for i, d in enumerate(data_chars):
        d = d[:max_length]
        d += [' '] * (max_length - len(d))
        data_chars[i] = d

    data_ids = np.zeros([len(data_chars), max_length], dtype=np.int64)
    for i in range(len(data_chars)):
        for j in range(max_length):
            data_ids[i, j] = char_to_ix[data_chars[i][j]]
    return np.array(data_ids)


def preprocess_word(contents, word_dict, document_max_len):
    x = map(lambda d: word_tokenize(d), contents)
    x = map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x)
    x = map(lambda d: d + [word_dict["<eos>"]], x)
    x = map(lambda d: d[:document_max_len], x)
    x = map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x)
    return list(x)


def read_data_chars():
    x_train, y_train, x_test, y_test = [], [], [], []
    with open(TRAIN_FILE, encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(clean_str(row[1]))
            y_train.append(int(row[0]))

    with open(TEST_FILE, encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(clean_str(row[1]))
            y_test.append(int(row[0]))

    vocab_size, char_to_ix = build_char_dict(x_train + x_test)
    x_train = preprocess_char(x_train, char_to_ix, MAX_DOCUMENT_LENGTH)
    y_train = np.array(y_train)
    x_test = preprocess_char(x_test, char_to_ix, MAX_DOCUMENT_LENGTH)
    y_test = np.array(y_test)

    x_train = tf.constant(x_train, dtype=tf.int64)
    y_train = tf.constant(y_train, dtype=tf.int64)
    x_test = tf.constant(x_test, dtype=tf.int64)
    y_test = tf.constant(y_test, dtype=tf.int64)
    logger.debug(f"shapes : x_train{x_train.shape}, y_train:{y_train.shape}, x_test:{x_test.shape}, y_test:{y_test.shape}")

    # Use `tf.data` to batch and shuffle the dataset:
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    return train_ds, test_ds


def read_data_words():
    x_train, y_train, x_test, y_test = [], [], [], []
    with open(TRAIN_FILE, encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(clean_str(row[1]))
            y_train.append(int(row[0]))

    with open(TEST_FILE, encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(clean_str(row[1]))
            y_test.append(int(row[0]))

    word_dict = build_word_dict(x_train + x_test)
    x_train = preprocess_word(x_train, word_dict, MAX_DOCUMENT_LENGTH)
    y_train = np.array(y_train)
    x_test = preprocess_word(x_test, word_dict, MAX_DOCUMENT_LENGTH)
    y_test = np.array(y_test)

    x_train = tf.constant(x_train, dtype=tf.int64)
    y_train = tf.constant(y_train, dtype=tf.int64)
    x_test = tf.constant(x_test, dtype=tf.int64)
    y_test = tf.constant(y_test, dtype=tf.int64)
    logger.debug(f"shapes : x_train{x_train.shape}, y_train:{y_train.shape}, x_test:{x_test.shape}, y_test:{y_test.shape}")

    vocab_size = tf.get_static_value(tf.reduce_max(x_train))
    vocab_size = max(vocab_size, tf.get_static_value(tf.reduce_max(x_test))) + 1

    # Use `tf.data` to batch and shuffle the dataset:
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
    return train_ds, test_ds, vocab_size
