import os
import numpy as np
from keras.utils import to_categorical
import sys
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import keras
from keras import layers


def read_yahoo_files():
    text_data_dir = '../../Yahoo/data/yahoo_10'
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    texts.append(f.read())
                    f.close()
                    labels.append(label_id)

    # labels = to_categorical(np.asarray(labels))
    return texts, labels, labels_index


def get_Yahoo_data():
    max_features = 20000
    max_len = 1000
    texts, labels, labels_index = read_yahoo_files()
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    labels = np.asarray(labels)
    return data, labels, texts


def Yahoo_LSTM():
    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.5)(x)
    # Add a classifier
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def Yahoo_LSTM_dropout():
    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.5)(x, training=True)
    # Add a classifier
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model

def Yahoo_GRU():
    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(64))(x)
    x = layers.Dropout(0.5)(x)
    # Add a classifier
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def Yahoo_GRU_dropout():
    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(64))(x)
    x = layers.Dropout(0.5)(x, training=True)
    # Add a classifier
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num", "-num",
                        type=int,
                        default=1
                        )
    parser.add_argument("--model_type", "-model_type",
                        type=str,
                        default='gru'
                        )
    args = parser.parse_args()
    num = args.num
    model_type = args.model_type
    # data, labels, texts = get_Yahoo_data()
    # train_index = np.load("../../Yahoo/data/train_indices.npy")
    # test_index = np.load("../../Yahoo/data/test_indices.npy")
    # x_train = data[train_index]
    # y_train = labels[train_index]
    # x_test = data[test_index]
    # y_test = labels[test_index]
    # print(len(y_test))
    #
    # y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)
    if model_type == 'lstm':
        model = Yahoo_LSTM()
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        # model.fit(x_train, y_train, batch_size=32, epochs=14, shuffle=True, validation_data=(x_test, y_test))
        # model.save("../../new_models/RQ1/Yahoo/lstm_" + str(num) + ".h5")
    elif model_type == 'gru':
        model = Yahoo_GRU()
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        # model.fit(x_train, y_train, batch_size=32, epochs=14, shuffle=True, validation_data=(x_test, y_test))
        # model.save("../../new_models/RQ1/Yahoo/gru_" + str(num) + ".h5")
#     a = np.arange(len(train_index))
#     print(len(train_index))
    # np.save("data/remain_data.npy", a)
    #
    # model = Yahoo_LSTM()
    # model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    # model.summary()
    # model.save("../new_models/RQ1/Yahoo/EGL_lstm_ori.h5")
    # model.fit(x_train, y_train, batch_size=32, epochs=14, shuffle=True, validation_data=(x_test, y_test))
    # model.save("models/yahoo_gru2.h5")


#   2858    57     4    23  3742    23    38 10315]
# [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
