# usage: python MNISTLeNet_5.py - train the model
from keras.utils import to_categorical
import numpy as np
from keras.preprocessing import text
import keras
from keras.datasets import mnist, cifar10
import csv
# from Yelp_model import *
from strategy import *
from QC_model import *

import pandas as pd
import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# tf.config.run_functions_eagerly(False)


def NiN_al(model, windows):
    data, label = get_QC_data("../../data/train_data_pytorch.csv", "../../data/test_data_pytorch.csv")
    train_indices = np.load("../../QC/data/training_indices.npy")
    test_indices = np.load("../../QC/data/test_indices.npy")
    target_data = data[train_indices]
    y_train = label[train_indices]
    x_test = data[test_indices]
    y_test = label[test_indices]
    y_train = to_categorical(y_train, 7)
    y_test = to_categorical(y_test, 7)

    all_index = np.array([i for i in range(20000)])
    remain_index = np.load("../data/QC_gru_remain_data.npy")
    selected_already = np.delete(all_index, remain_index)
    target_data = target_data[remain_index]
    y_train = y_train[remain_index]
    selected_index = EGL_selection_index(model, target_data, y_train, windows)
    print("#######################")
    print(selected_index)
    print("#######################")

    now_select = np.append(selected_already, remain_index[selected_index])
    now_remain = np.delete(remain_index, selected_index)
    np.save("../data/QC_gru_selected_index.npy", now_select)
    np.save("../data/QC_gru_remain_data.npy", now_remain)


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
        windows = 1000
        epochs = 10
        threshold = 0.87

        # results_path = args.results
        # model_save_path = args.model
        # results_path = "results/al_results_kcenter.csv"
        # model_save_path = "models/kcenter_lenet_5.h5"
        # model = Yelp_gru()
        # model.save("Yelp_init.h5")
        model = keras.models.load_model("../../new_models/RQ1/QC/EGL_gru_ori.h5")
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # dropout_model = Yelp_gru_dropout()
        # dropout_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        NiN_al(model, windows)

    # python -u active_learning.py -metric 4 -results results/al_results_lc.csv -model models/lc_lenet_5.h5






