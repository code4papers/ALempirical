from keras.datasets import cifar10, mnist
from keras.models import load_model
import numpy as np
import keras
from art.metrics.metrics import empirical_robustness, clever_t, clever_u, clever, loss_sensitivity, wasserstein_distance
from art.estimators.classification.keras import KerasClassifier
import tensorflow as tf
import argparse
import csv
import keras.backend as K
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, SaliencyMapMethod, CarliniL2Method
from keras.datasets import mnist
from QC.QC_model import *
from Yahoo.Yahoo_model import *
tf.compat.v1.experimental.output_all_intermediates(True)
tf.compat.v1.disable_eager_execution()
# tf.compat.v1.experimental.output_all_intermediates(True)


def CLEVER_metric(model, data, norm, nb_batches, batch_size, radius):
    total_res = 0
    for _ in range(len(data)):
        print(_)
        res0 = clever_u(model, data[_], nb_batches, batch_size, radius, norm=norm, pool_factor=3)
        total_res += res0
    total_res = total_res / len(data)
    return total_res


def empirical_robustness_metric(model, data):
    params = {"eps_step": 0.1, "eps": 0.3}
    emp_robust1 = empirical_robustness(model, data, str("fgsm"), params)
    return emp_robust1


def loss_sensitivity_metric(model, data, label):
    loss1 = loss_sensitivity(model, data, label)
    return loss1


def imdb_model_without_embedding():
    inputs = keras.Input(shape=(200, 128))

    x = keras.layers.SpatialDropout1D(0.2)(inputs)
    x = keras.layers.LSTM(128)(x)

    x = keras.layers.Dropout(0.2)(x)
    # Add a classifier
    x = keras.layers.Dense(20, activation="relu")(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model


def imdb_model_gru_without_embedding():
    inputs = keras.Input(shape=(200, 128))

    x = keras.layers.SpatialDropout1D(0.2)(inputs)
    x = keras.layers.GRU(128)(x)

    x = keras.layers.Dropout(0.2)(x)
    # Add a classifier
    x = keras.layers.Dense(20, activation="relu")(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model


def qc_model_without_embedding():
    inputs = keras.Input(shape=(100, 100))
    x = keras.layers.Dropout(0.2)(inputs)
    x = keras.layers.Conv1D(128, 5, activation='relu')(x)
    x = keras.layers.MaxPool1D(2)(x)
    x = keras.layers.LSTM(128)(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(7, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def qc_model_gru_without_embedding():
    inputs = keras.Input(shape=(100, 100))
    x = keras.layers.Dropout(0.2)(inputs)
    x = keras.layers.Conv1D(128, 5, activation='relu')(x)
    x = keras.layers.MaxPool1D(2)(x)
    x = keras.layers.GRU(128)(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(7, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def yahoo_model_lstm_without_embedding():
    # max_features = 20000  # Only consider the top 20k words
    inputs = keras.Input(shape=(1000, 128))
    # Input for variable-length sequences of integers
    # Add 2 bidirectional LSTMs
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64))(inputs)
    x = keras.layers.Dropout(0.5)(x)
    # Add a classifier
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def yahoo_model_gru_without_embedding():
    # max_features = 20000  # Only consider the top 20k words
    inputs = keras.Input(shape=(1000, 128))
    # Input for variable-length sequences of integers
    # Add 2 bidirectional LSTMs
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = keras.layers.Bidirectional(keras.layers.GRU(64))(inputs)
    x = keras.layers.Dropout(0.5)(x)
    # Add a classifier
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def IMDB_CLEVER(model_path, select_data, results_path, norm):
    ori_model = load_model(model_path)
    get_3rd_layer_output = K.function([ori_model.layers[0].input],
                                      [ori_model.layers[1].output])
    max_features = 20000
    maxlen = 200
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
        num_words=max_features
    )

    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
    select_indices = np.load(select_data)
    test_data = get_3rd_layer_output([x_val[select_indices]])[0]
    print(test_data.shape)
    # model = imdb_model_gru_without_embedding()
    if "lstm" in model_path:
        model = imdb_model_without_embedding()
    else:
        model = imdb_model_gru_without_embedding()
    layer_len = len(ori_model.layers)
    for i in range(2, layer_len - 1):
        model.layers[i].set_weights(ori_model.layers[i + 1].get_weights())
    # print(model.predict(test_data))
    krc = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
    CLEVER = CLEVER_metric(krc, test_data, norm, 10, 10, 5)
    # er = empirical_robustness_metric(krc, test_data)
    print(CLEVER)
    csv_file = open(results_path, "a")
    try:
        writer = csv.writer(csv_file)
        writer.writerow([model_path, CLEVER])

    finally:
        csv_file.close()


def QC_CLEVER(model_path, select_data, results_path, norm):
    ori_model = load_model(model_path)
    get_3rd_layer_output = K.function([ori_model.layers[0].input],
                                      [ori_model.layers[1].output])
    data, label = get_QC_data("../../data/train_data_pytorch.csv", "../../data/test_data_pytorch.csv")
    test_indices = np.load("../../QC/data/test_indices.npy")
    x_test = data[test_indices]

    select_indices = np.load(select_data)
    test_data = get_3rd_layer_output([x_test[select_indices]])[0]
    print(test_data.shape)
    if "lstm" in model_path:
        model = qc_model_without_embedding()
    else:
        model = qc_model_gru_without_embedding()
    layer_len = len(ori_model.layers)
    for i in range(2, layer_len - 1):
        model.layers[i].set_weights(ori_model.layers[i + 1].get_weights())
    # print(model.predict(test_data))
    krc = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
    CLEVER = CLEVER_metric(krc, test_data, norm, 10, 10, 5)
    # er = empirical_robustness_metric(krc, test_data)
    print(CLEVER)
    csv_file = open(results_path, "a")
    try:
        writer = csv.writer(csv_file)
        writer.writerow([model_path, CLEVER])

    finally:
        csv_file.close()


def Yahoo_CLEVER(model_path, results_path, norm):
    ori_model = load_model(model_path)
    get_3rd_layer_output = K.function([ori_model.layers[0].input],
                                      [ori_model.layers[1].output])
    data, labels, texts = get_Yahoo_data()
    test_index = np.load("../../Yahoo/data/test_indices.npy")
    x_test = data[test_index]
    # select_indices = np.load(select_data)
    test_data = get_3rd_layer_output([x_test[:2]])[0]
    print(test_data.shape)
    if "lstm" in model_path:
        model = yahoo_model_lstm_without_embedding()
    else:
        model = yahoo_model_gru_without_embedding()
    layer_len = len(ori_model.layers)
    for i in range(2, layer_len - 1):
        model.layers[i].set_weights(ori_model.layers[i + 1].get_weights())
    # print(model.predict(test_data))
    krc = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
    CLEVER = CLEVER_metric(krc, test_data, norm, 10, 10, 5)
    # er = empirical_robustness_metric(krc, test_data)
    print(CLEVER)
    csv_file = open(results_path, "a")
    try:
        writer = csv.writer(csv_file)
        writer.writerow([model_path, CLEVER])

    finally:
        csv_file.close()


def MNIST_CLEVER(model_path, select_data, results_path, norm):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    model = load_model(model_path)
    select_indices = np.load(select_data)
    krc = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
    CLEVER = CLEVER_metric(krc, x_test[select_indices], norm, 10, 10, 5)
    print(CLEVER)
    csv_file = open(results_path, "a")
    try:
        writer = csv.writer(csv_file)
        writer.writerow([model_path, CLEVER])

    finally:
        csv_file.close()


def NiN_CLEVER(model_path, select_data, results_path, norm):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    model = load_model(model_path)
    select_indices = np.load(select_data)
    krc = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
    CLEVER = CLEVER_metric(krc, x_test[select_indices], norm, 10, 10, 5)
    print(CLEVER)
    csv_file = open(results_path, "a")
    try:
        writer = csv.writer(csv_file)
        writer.writerow([model_path, CLEVER])

    finally:
        csv_file.close()


def VGG16_CLEVER(model_path, select_data, results_path, norm):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    model = load_model(model_path)
    select_indices = np.load(select_data)
    krc = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
    CLEVER = CLEVER_metric(krc, x_test[select_indices], norm, 10, 10, 5)
    print(CLEVER)
    csv_file = open(results_path, "a")
    try:
        writer = csv.writer(csv_file)
        writer.writerow([model_path, CLEVER])

    finally:
        csv_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-dataset",
                        type=str,
                        )
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        )
    parser.add_argument("--data_path", "-data_path",
                        type=str,
                        )
    parser.add_argument("--results_path", "-results_path",
                        type=str,
                        )
    parser.add_argument("--norm", "-norm",
                        type=str,
                        )

    # model = load_model("NiN/NiNmodels/NiN.h5")
    args = parser.parse_args()
    data_type = args.dataset
    model_path = args.model_path
    data_path = args.data_path
    results_path = args.results_path
    norm = args.norm
    if norm == '1':
        norm = 1
    elif norm == '2':
        norm = 2
    elif norm == 'inf':
        norm = np.inf
    # if data_type == 0:
    #     IMDB_CLEVER(model_path, data_path, results_path)
    # elif data_type == 1:
    #     QC_CLEVER(model_path, data_path, results_path)
    # else:
    #     Yahoo_CLEVER(model_path, results_path)
    if data_type == 'mnist':
        MNIST_CLEVER(model_path, data_path, results_path, norm)
    if data_type == 'NiN':
        NiN_CLEVER(model_path, data_path, results_path, norm)
    if data_type == 'VGG16':
        VGG16_CLEVER(model_path, data_path, results_path, norm)
    if data_type == 'qc':
        QC_CLEVER(model_path, data_path, results_path, norm)
    if data_type == 'yahoo':
        Yahoo_CLEVER(model_path, results_path, norm)
    if data_type == 'imdb':
        IMDB_CLEVER(model_path, data_path, results_path, norm)




# ori_model = load_model("QC/QCmodels/QC_lstm.h5")
# model = load_model("QC/QCmodels/QC_lstm.h5")
#
# ori_model.summary()
# get_3rd_layer_output = K.function([ori_model.layers[0].input],
#                                   [ori_model.layers[1].output])

# IMDB
# max_features = 20000
# maxlen = 200
# (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
#     num_words=max_features
# )
# print(len(x_train), "Training sequences")
# print(len(x_val), "Validation sequences")
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
#
# test_data = get_3rd_layer_output([x_val[:5]])[0]
# print(test_data.shape)
# model = imdb_model_without_embedding()
# layer_len = len(ori_model.layers)
# for i in range(2, layer_len - 1):
#     print(model.layers[i])
#     print(ori_model.layers[i + 1])
#     model.layers[i].set_weights(ori_model.layers[i + 1].get_weights())
# # print(model.predict(test_data))
# krc = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
# CLEVER = CLEVER_metric(krc, test_data, 10, 10, 5)
# # er = empirical_robustness_metric(krc, test_data)
# print(CLEVER)

# QC
###########################################################
# data, label = get_QC_data("data/train_data_pytorch.csv", "data/test_data_pytorch.csv")
# train_indices = np.load("QC/data/training_indices.npy")
# test_indices = np.load("QC/data/test_indices.npy")
# x_train = data[train_indices]
# y_train = label[train_indices]
# x_test = data[test_indices]
# y_test = label[test_indices]
# y_train = to_categorical(y_train, 7)
# y_test = to_categorical(y_test, 7)
# test_data = get_3rd_layer_output([x_test[:5]])[0]
# new_model = qc_model_without_embedding()
# layer_len = len(ori_model.layers)
# for i in range(2, layer_len - 1):
#     new_model.layers[i].set_weights(ori_model.layers[i + 1].get_weights())
# print(test_data[0].shape)
# krc = KerasClassifier(model=new_model, clip_values=(0, 1), use_logits=False)
# CLEVER = CLEVER_metric(krc, test_data, 10, 10, 5)
# print(CLEVER)


