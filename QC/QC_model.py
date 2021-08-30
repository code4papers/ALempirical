from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dropout, Dense, Input, GRU
import os
import keras
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np


def get_QC_data(train_data_path, test_data_path):
    max_features = 10000
    max_len = 100
    texts = []
    labels = []
    qc_train = pd.read_csv(train_data_path,
                     names=['num', 'title', 'description', 'category'])
    # print(qc['description'][0])
    for i in range(len(qc_train)):
        texts.append(str(qc_train['description'][i]))
        labels.append(qc_train['category'][i])

    qc_test = pd.read_csv(test_data_path,
                          names=['num', 'title', 'description', 'category'])
    # print(qc['description'][0])
    for i in range(len(qc_test)):
        texts.append(str(qc_test['description'][i]))
        labels.append(qc_test['category'][i])

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_len)
    labels = np.asarray(labels)
    return data, labels


def QC_LSTM(emb_train=True):

    max_features = 10000
    inputs = Input(shape=(100,), dtype="int32")
    if emb_train:
        x = Embedding(max_features, 100)(inputs)
    else:
        x = Embedding(max_features, 100, trainable=False)(inputs)
    x = Dropout(0.2)(x)
    x = keras.layers.Conv1D(128, 5, activation='relu')(x)
    x = keras.layers.MaxPool1D(2)(x)
    x = LSTM(128, recurrent_dropout=0.2)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(7, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def QC_LSTM_dropout(emb_train=True):

    max_features = 10000
    inputs = Input(shape=(100,), dtype="int32")
    if emb_train:
        x = Embedding(max_features, 100)(inputs)
    else:
        x = Embedding(max_features, 100, trainable=False)(inputs)
    x = Dropout(0.2)(x)
    x = keras.layers.Conv1D(128, 5, activation='relu')(x)
    x = keras.layers.MaxPool1D(2)(x)
    x = LSTM(128, recurrent_dropout=0.2)(x)
    x = Dropout(0.2)(x, training=True)
    outputs = Dense(7, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def QC_GRU(emb_train=True):

    max_features = 10000
    inputs = Input(shape=(100,), dtype="int32")
    if emb_train:
        x = Embedding(max_features, 100)(inputs)
    else:
        x = Embedding(max_features, 100, trainable=False)(inputs)
    x = Dropout(0.2)(x)
    x = keras.layers.Conv1D(128, 5, activation='relu')(x)
    x = keras.layers.MaxPool1D(2)(x)
    x = GRU(128, recurrent_dropout=0.2)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(7, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def QC_GRU_dropout(emb_train=True):

    max_features = 10000
    inputs = Input(shape=(100,), dtype="int32")
    if emb_train:
        x = Embedding(max_features, 100)(inputs)
    else:
        x = Embedding(max_features, 100, trainable=False)(inputs)
    x = Dropout(0.2)(x)
    x = keras.layers.Conv1D(128, 5, activation='relu')(x)
    x = keras.layers.MaxPool1D(2)(x)
    x = GRU(128, recurrent_dropout=0.2)(x)
    x = Dropout(0.2)(x, training=True)
    outputs = Dense(7, activation='softmax')(x)
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

                        default = 'gru'
                        )
    args = parser.parse_args()
    num = args.num
    model_type = args.model_type
    # train_indices = np.load("../../QC/data/training_indices.npy")
    # test_indices = np.load("../../QC/data/test_indices.npy")
    # data, label = get_QC_data("../../data/train_data_pytorch.csv", "../../data/test_data_pytorch.csv")
    # x_train = data[train_indices]
    # y_train = label[train_indices]
    # x_test = data[test_indices]
    # y_test = label[test_indices]

    # y_train = to_categorical(y_train, 7)
    # y_test = to_categorical(y_test, 7)
    if model_type == 'lstm':
        model = QC_LSTM()
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        # his = model.fit(x_train, y_train, batch_size=128, epochs=10, shuffle=True, validation_data=(x_test, y_test))
        # model.save("../../new_models/RQ1/QC/lstm_" + str(num) + ".h5")
    elif model_type == 'gru':
        model = QC_GRU()
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        # his = model.fit(x_train, y_train, batch_size=128, epochs=10, shuffle=True, validation_data=(x_test, y_test))
        # model.save("../../new_models/RQ1/QC/gru_" + str(num) + ".h5")
# # model = QC_GRU(emb_train=True)
# # model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
# # ori_model = load_model("QCmodels/QC_lstm.h5")
# # model.layers[1].set_weights(ori_model.layers[1].get_weights())
# # model.save("QCmodels/EGL_QC_lstm.h5")
# data, label = get_QC_data("../data/train_data_pytorch.csv", "../data/test_data_pytorch.csv")
# # all_indices = np.arange(len(data))
# # test_indices = np.random.choice(len(data), 2000, replace=False)
# # remain_indices = np.delete(all_indices, test_indices)
# # train_indices = np.random.choice(len(remain_indices), 20000, replace=False)
# # np.save("data/training_indices.npy", train_indices)
# # np.save("data/test_indices.npy", test_indices)
# train_indices = np.load("data/training_indices.npy")
# test_indices = np.load("data/test_indices.npy")
#
# # include_idx = set(indices_select)  #Set is more efficient, but doesn't reorder your elements if that is desireable
# # mask = np.array([(i in include_idx) for i in range(len(data))])
# # x_train = data[~mask][:20000]
# # y_train = label[~mask][:20000]
# # print(~mask)
# # x_test = data[mask]
# # y_test = label[mask]
#
# x_train = data[train_indices]
# y_train = label[train_indices]
# x_test = data[test_indices]
# y_test = label[test_indices]
#
# y_train = to_categorical(y_train, 7)
# y_test = to_categorical(y_test, 7)
#
# print(x_train.shape)
# print(x_test.shape)
# model = QC_GRU()
# model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
# model.save("../new_models/RQ1/QC/EGL_gru_ori.h5")
# model.summary()
# his = model.fit(x_train, y_train, batch_size=128, epochs=10, shuffle=True, validation_data=(x_test, y_test))
# print(his.history['val_accuracy'])
# model.save("QCmodels/QC_gru.h5")


