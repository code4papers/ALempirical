# usage: python MNISTLeNet_5.py - train the model
from keras.utils import to_categorical
import numpy as np
from keras.preprocessing import text
import keras
from keras.datasets import mnist, cifar10, cifar100
import csv
# from Yelp_model import *
from strategy import *
import pandas as pd
import argparse
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# tf.config.run_functions_eagerly(False)
def color_preprocessing(x_train, x_test):

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def wideresnet_al(model, windows):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    # Normalize data.
    target_data = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    target_data, x_test = color_preprocessing(target_data, x_test)

    # If subtract pixel mean is enabled
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)
    all_index = np.array([i for i in range(50000)])
    remain_index = np.load("WideResNet/data/remain_data.npy")
    selected_already = np.delete(all_index, remain_index)
    target_data = target_data[remain_index]
    y_train = y_train[remain_index]
    selected_index = EGL_selection_index(model, target_data, y_train, windows)
    print("#######################")
    print(selected_index)
    print("#######################")

    now_select = np.append(selected_already, remain_index[selected_index])
    now_remain = np.delete(remain_index, selected_index)
    np.save("WideResNet/data/selected_index.npy", now_select)
    np.save("WideResNet/data/remain_data.npy", now_remain)


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
        windows = 2500
        # epochs = 5
        threshold = 0.7
        # results_path = args.results
        # model_save_path = args.model
        # results_path = "results/al_results_kcenter.csv"
        # model_save_path = "models/kcenter_lenet_5.h5"
        # model = Yelp_LSTM()
        # model.save("Yelp_init.h5")
        model = keras.models.load_model("WideResNet/models/EGL_wideresnet.h5")
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # dropout_model = Yelp_LSTM_dropout()
        # dropout_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        wideresnet_al(model, windows)

    # python -u active_learning.py -metric 4 -results results/al_results_lc.csv -model models/lc_lenet_5.h5






