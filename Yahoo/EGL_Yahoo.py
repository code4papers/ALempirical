# usage: python MNISTLeNet_5.py - train the model
from keras.utils import to_categorical
import numpy as np
from keras.preprocessing import text
import keras
from keras.datasets import mnist, cifar10
import csv
# from Yelp_model import *
from strategy import *
from Yahoo_model import *

import pandas as pd
import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# tf.config.run_functions_eagerly(False)


def NiN_al(model, windows):
    data, labels, texts = get_Yahoo_data()
    train_index = np.load("../../Yahoo/data/train_indices.npy")
    test_index = np.load("../../Yahoo/data/test_indices.npy")
    target_data = data[train_index]
    y_train = labels[train_index]
    x_test = data[test_index]
    y_test = labels[test_index]

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    all_index = np.array([i for i in range(len(train_index))])
    remain_index = np.load("../data/Yahoo_lstm_remain_data.npy")
    selected_already = np.delete(all_index, remain_index)
    target_data = target_data[remain_index]
    y_train = y_train[remain_index]
    selected_index = EGL_selection_index(model, target_data, y_train, windows)
    print("#######################")
    print(selected_index)
    print("#######################")

    now_select = np.append(selected_already, remain_index[selected_index])
    now_remain = np.delete(remain_index, selected_index)
    np.save("../data/Yahoo_lstm_selected_index.npy", now_select)
    np.save("../data/Yahoo_lstm_remain_data.npy", now_remain)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--metric", "-metric",
    #                     type=int,
    #                     )
    # parser.add_argument("--results", "-results",
    #                     type=str,
    #                     )
    # parser.add_argument("--model", "-model",
    #                     type=str,
    #                     )
    # args = parser.parse_args()
    # metric = args.metric
    # results_path = args.results
    # model_save_path = args.model
    mode = 0
    # MNIST
    if mode == 0:
        steps = 1000
        windows = 178
        epochs = 16
        threshold = 0.88

        # results_path = args.results
        # model_save_path = args.model
        # results_path = "results/al_results_kcenter.csv"
        # model_save_path = "models/kcenter_lenet_5.h5"
        # model = Yelp_LSTM()
        # model.save("Yelp_init.h5")
        model = keras.models.load_model("../../new_models/RQ1/Yahoo/EGL_lstm_ori.h5")
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # dropout_model = Yelp_LSTM_dropout()
        # dropout_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        NiN_al(model, windows)

    # python -u active_learning.py -metric 4 -results results/al_results_lc.csv -model models/lc_lenet_5.h5






