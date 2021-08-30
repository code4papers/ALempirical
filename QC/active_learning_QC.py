# usage: python MNISTLeNet_5.py - train the model
from keras.utils import to_categorical
import numpy as np
from keras.datasets import mnist, cifar10
import csv
from .QC_model import *
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# from strategy_binary import *
from strategy import *
import argparse


def QC_al(model, steps, windows, epochs, results_path, strategy, model_save, dropout_model=None):
    data, label = get_QC_data("../../data/train_data_pytorch.csv", "../../data/test_data_pytorch.csv")
    train_indices = np.load("../../QC/data/training_indices.npy")
    test_indices = np.load("../../QC/data/test_indices.npy")
    target_data = data[train_indices]
    y_train = label[train_indices]
    x_test = data[test_indices]
    y_test = label[test_indices]
    y_train = to_categorical(y_train, 7)
    y_test = to_categorical(y_test, 7)
    # print(y_train[0])
    idxs_lb = np.zeros(len(target_data), dtype=bool)
    for _ in range(steps):
        if strategy == 0:
            selected_data, selected_label, remain_data, remain_label = entropy_selection(model, target_data, y_train,
                                                                                         windows)
            target_data = remain_data
            y_train = remain_label
        elif strategy == 1:
            model_weights = model.get_weights()
            dropout_model.set_weights(model_weights)
            selected_data, selected_label, remain_data, remain_label = BALD_selection(dropout_model, target_data, y_train,
                                                                                      windows)
            target_data = remain_data
            y_train = remain_label
        elif strategy == 2:
            model_weights = model.get_weights()

            dropout_model.set_weights(model_weights)
            selected_data, selected_label, remain_data, remain_label = entropy_dropout_selection(dropout_model, target_data, y_train,
                                                                                                 windows)
            target_data = remain_data
            y_train = remain_label
        elif strategy == 4:
            selected_data, selected_label, remain_data, remain_label = EGL_selection(model, target_data, y_train, windows)
            target_data = remain_data
            y_train = remain_label
        elif strategy == 5:

            selected_data, selected_label, remain_data, remain_label = margin_selection(model, target_data, y_train,
                                                                                        windows)

            target_data = remain_data

            y_train = remain_label

        elif strategy == 6:

            model_weights = model.get_weights()

            dropout_model.set_weights(model_weights)

            selected_data, selected_label, remain_data, remain_label = margin_dropout_selection(dropout_model,
                                                                                                 target_data, y_train,
                                                                                                 windows)

            target_data = remain_data

            y_train = remain_label

        elif strategy == 10:
            selected_data, selected_label, remain_data, remain_label = random_selection(model, target_data, y_train,
                                                                                     windows)
            target_data = remain_data
            y_train = remain_label


        if _ == 0:
            if strategy == 3:
                init_index = np.random.choice(len(target_data), windows)
                idxs_lb[init_index] = True
                training_data = target_data[init_index]
                training_label = y_train[init_index]

            else:
                training_data = selected_data
                training_label = selected_label
        else:
            if strategy == 3:
                selected_data, selected_label, remain_data, remain_label, idxs_lb = k_center_greedy_selection(model,
                                                                                                              target_data,
                                                                                                              y_train,
                                                                                                              windows,
                                                                                                              idxs_lb)

            training_data = np.concatenate((training_data, selected_data))
            training_label = np.concatenate((training_label, selected_label))

        train_len = len(training_data)
        print("training data len: {}".format(train_len))
        check_point = ModelCheckpoint(model_save, monitor="val_accuracy", save_best_only=True, verbose=1)
        # his = model.fit(training_data, training_label, batch_size=32, shuffle=True, epochs=epochs,
        #                 validation_data=(x_val, y_val), verbose=1)
        his = model.fit(training_data, training_label, batch_size=128,
                        epochs=epochs, validation_data=(x_test, y_test),
                        callbacks=[check_point])

        val_acc = np.max(his.history['val_accuracy'])
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([train_len, val_acc])

        finally:
            csv_file.close()
        # if val_acc > threshold or train_len >= 10000:
        #     break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", "-metric",
                        type=int,
                        )
    parser.add_argument("--results", "-results",
                        type=str,
                        )
    parser.add_argument("--model", "-model",
                        type=str,
                        )
    args = parser.parse_args()
    metric = args.metric
    results_path = args.results
    model_save_path = args.model
    mode = 0
    # MNIST
    if mode == 0:
        steps = 1000
        windows = 200
        epochs = 10
        threshold = 0.87
        results_path = args.results
        model_save_path = args.model
        # results_path = "results/al_results_kcenter.csv"
        # model_save_path = "models/kcenter_lenet_5.h5"
        #########################################################
        # IMDB old
        #########################################################
        # model = IMDB_GRU()
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # dropout_model = IMDB_GRU_dropout()
        # dropout_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        #########################################################
        # IMDB new
        #########################################################
        model = QC_GRU(emb_train=True)
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        # ori_model = load_model("QCmodels/QC_lstm.h5")
        # model.layers[1].set_weights(ori_model.layers[1].get_weights())
        dropout_model = QC_GRU_dropout()
        dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        QC_al(model, steps, windows, epochs, results_path, metric, model_save_path, dropout_model)


    # python -u active_learning.py -metric 4 -results results/al_results_lc.csv -model models/lc_lenet_5.h5






