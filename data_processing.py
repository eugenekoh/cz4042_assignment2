import collections
import csv
import re
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from loguru import logger
from nltk.tokenize import word_tokenize

from constants import TRAIN_FILE, TEST_FILE, BATCH_SIZE, MAX_DOCUMENT_LENGTH


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\" ]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text


class DataProcessor(ABC):
    @abstractmethod
    def build_dict(self, strings):
        pass

    @abstractmethod
    def process(self, strings, data_dict, max_length):
        pass


class CharProcessor(DataProcessor):
    def build_dict(self, strings):
        chars = sorted(set(''.join(strings)))
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        vocab_size = len(chars)
        return vocab_size, char_to_ix

    def process(self, strings, data_dict, max_length):
        data_chars = [list(text) for text in strings]
        for i, d in enumerate(data_chars):
            d = d[:max_length]
            d += [' '] * (max_length - len(d))
            data_chars[i] = d

        data_ids = np.zeros([len(data_chars), max_length], dtype=np.int64)
        for i in range(len(data_chars)):
            for j in range(max_length):
                data_ids[i, j] = data_dict[data_chars[i][j]]
        return np.array(data_ids)


class WordProcessor(DataProcessor):
    def build_dict(self, strings):
        words = list()
        for content in strings:
            for word in word_tokenize(clean_str(content)):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<eos>"] = 2
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)
        vocab_size = len(word_dict)
        return vocab_size, word_dict

    def process(self, strings, data_dict, max_length):
        x = map(lambda d: word_tokenize(d), strings)
        x = map(lambda d: list(map(lambda w: data_dict.get(w, data_dict["<unk>"]), d)), x)
        x = map(lambda d: d + [data_dict["<eos>"]], x)
        x = map(lambda d: d[:max_length], x)
        x = map(lambda d: d + (max_length - len(d)) * [data_dict["<pad>"]], x)
        return np.asarray(list(x))


def read_data(processor):
    assert (isinstance(processor, DataProcessor))

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

    vocab_size, data_dict = processor.build_dict(x_train + x_test)
    x_train = processor.process(x_train, data_dict, MAX_DOCUMENT_LENGTH)
    y_train = np.array(y_train)
    x_test = processor.process(x_test, data_dict, MAX_DOCUMENT_LENGTH)
    y_test = np.array(y_test)

    x_train = tf.constant(x_train, dtype=tf.int64)
    y_train = tf.constant(y_train, dtype=tf.int64)
    x_test = tf.constant(x_test, dtype=tf.int64)
    y_test = tf.constant(y_test, dtype=tf.int64)
    logger.debug(
        f"shapes - x_train:{x_train.shape}, y_train:{y_train.shape}, x_test:{x_test.shape}, y_test:{y_test.shape}, "
        f"vocab_size: {vocab_size}")

    # Use `tf.data` to batch and shuffle the dataset:
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
    return train_ds, test_ds, vocab_size


def read_data_chars():
    return read_data(CharProcessor())


def read_data_words():
    return read_data(WordProcessor())
