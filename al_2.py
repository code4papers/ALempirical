# usage: python MNISTLeNet_5.py - train the model

from __future__ import print_function
from keras.utils import to_categorical
import numpy as np
from strategy import *
from Lenet_5 import *
from keras.datasets import mnist, cifar10
from resnet20 import *
from NiN.NiNmodel import *
import csv


def MNIST_al(model, steps, windows, epochs, threshold, results_path, strategy, model_save, dropout_model=None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    target_data = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_test = to_categorical(y_test, 10)
    y_train = to_categorical(y_train, 10)
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
            selected_data = selected_data.reshape(-1, 28, 28, 1)
            target_data = remain_data
            y_train = remain_label
        elif strategy == 2:
            selected_data, selected_label, remain_data, remain_label = Kmeans_selection(model, target_data, y_train,
                                                                                         windows)
            target_data = remain_data
            y_train = remain_label
        elif strategy == 5:
            selected_data, selected_label, remain_data, remain_label = least_confidence(model, target_data, y_train, windows)
            target_data = remain_data
            y_train = remain_label
        elif strategy == 6:
            selected_data, selected_label, remain_data, remain_label = margin_selection(model, target_data, y_train, windows)
            target_data = remain_data
            y_train = remain_label
        elif strategy == 7:
            model_weights = model.get_weights()
            dropout_model.set_weights(model_weights)
            selected_data, selected_label, remain_data, remain_label = entropy_dropout_selection(dropout_model, target_data, y_train, windows)
            target_data = remain_data
            y_train = remain_label
        elif strategy == 8:
            model_weights = model.get_weights()
            dropout_model.set_weights(model_weights)
            selected_data, selected_label, remain_data, remain_label = margin_dropout_selection(dropout_model, target_data, y_train, windows)
            target_data = remain_data
            y_train = remain_label
        elif strategy == 9:
            model_weights = model.get_weights()
            dropout_model.set_weights(model_weights)
            selected_data, selected_label, remain_data, remain_label = least_confidence_dropout_selection(dropout_model, target_data, y_train, windows)
            target_data = remain_data
            y_train = remain_label
        if _ == 0:
            if strategy == 3 or strategy == 4:
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
            elif strategy == 4:
                selected_data, selected_label, remain_data, remain_label, idxs_lb = coreset_selection(model,
                                                                                                      target_data,
                                                                                                      y_train,
                                                                                                      windows,
                                                                                                      idxs_lb)
                # selected_data, selected_label, remain_data, remain_label, trained_data, trained_label = coreset_selection_2(model, trained_data, trained_label, trained_data, trained_label, windows)
                # target_data = remain_data
                # y_train = remain_label
            training_data = np.concatenate((training_data, selected_data))
            training_label = np.concatenate((training_label, selected_label))

        train_len = len(training_data)
        print("training data len: {}".format(train_len))
        his = model.fit(training_data, training_label, batch_size=256, shuffle=True, epochs=epochs,
                        validation_data=(x_test, y_test), verbose=1)
        val_acc = his.history['val_accuracy'][-1]
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([train_len, val_acc])

        finally:
            csv_file.close()
        if val_acc > threshold:
            model.save(model_save)
            break


def Cifar10_al(model, steps,  windows, epochs, threshold, results_path, metric, model_save_path, dropout_model=None, data_aug=True):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalize data.
    target_data = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    x_train_mean = np.mean(target_data, axis=0)
    target_data -= x_train_mean
    x_test -= x_train_mean
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    idxs_lb = np.zeros(len(target_data), dtype=bool)

    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    for _ in range(steps):
        if metric == 0:
            selected_data, selected_label, remain_data, remain_label = entropy_selection(model, target_data, y_train,
                                                                                         windows)
            target_data = remain_data
            y_train = remain_label

        elif metric == 1:
            model_weights = model.get_weights()
            dropout_model.set_weights(model_weights)
            selected_data, selected_label, remain_data, remain_label = BALD_selection(dropout_model, target_data,
                                                                                      y_train,
                                                                                      windows)
            selected_data = selected_data.reshape(-1, 32, 32, 3)
            target_data = remain_data
            y_train = remain_label

        elif metric == 6:

            selected_data, selected_label, remain_data, remain_label = margin_selection(model, target_data, y_train,
                                                                                        windows)

            target_data = remain_data

            y_train = remain_label

        elif metric == 7:

            model_weights = model.get_weights()

            dropout_model.set_weights(model_weights)

            selected_data, selected_label, remain_data, remain_label = entropy_dropout_selection(dropout_model,
                                                                                                 target_data, y_train,
                                                                                                 windows)

            target_data = remain_data

            y_train = remain_label

        elif metric == 8:

            model_weights = model.get_weights()

            dropout_model.set_weights(model_weights)

            selected_data, selected_label, remain_data, remain_label = margin_dropout_selection(dropout_model,
                                                                                                target_data, y_train,
                                                                                                windows)

            target_data = remain_data

            y_train = remain_label

        if _ == 0:
            if metric == 3 or metric == 4:
                init_index = np.random.choice(len(target_data), windows)
                idxs_lb[init_index] = True
                training_data = target_data[init_index]
                training_label = y_train[init_index]

            else:
                training_data = selected_data
                training_label = selected_label
        else:
            if metric == 3:
                selected_data, selected_label, remain_data, remain_label, idxs_lb = k_center_greedy_selection(model,
                                                                                                              target_data,
                                                                                                              y_train,
                                                                                                              windows,
                                                                                                              idxs_lb)

            training_data = np.concatenate((training_data, selected_data))
            training_label = np.concatenate((training_label, selected_label))
        train_len = len(training_data)

        print("training data len: {}".format(train_len))

        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
        callbacks = [lr_reducer, lr_scheduler, early_stop]
        if data_aug:
            datagen.fit(training_data)
            # his = model.fit(training_data, training_label,
            #       batch_size=128,
            #       epochs=epochs,
            #       validation_data=(x_test, y_test),
            #       shuffle=True,
            #       callbacks=callbacks)
            his = model.fit_generator(datagen.flow(training_data, training_label, batch_size=128),
                                      validation_data=(x_test, y_test),
                                      epochs=epochs, verbose=0, workers=4,
                                      callbacks=callbacks)
        else:
            early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
            # checkpoint = ModelCheckpoint(model_save_path,
            #                              monitor='val_accuracy',
            #                              verbose=1,
            #                              save_best_only=True,
            #                              mode='max',
            #                              period=1)
            cbks = [early_stop]
            his = model.fit(x_train, y_train,
                            batch_size=128,
                            epochs=200,
                            callbacks=cbks,
                            validation_data=(x_test, y_test),
                            verbose=1)
        val_acc = his.history['val_accuracy'][-1]
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([train_len, val_acc])

        finally:
            csv_file.close()
        if val_acc > threshold or train_len >= 25000:
            model.save(model_save_path)
            break


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
    mode = 1
    # MNIST
    if mode == 0:
        steps = 1000
        windows = 100
        epochs = 5
        threshold = 0.985
        results_path = args.results
        model_save_path = args.model
        # results_path = "results/al_results_kcenter.csv"
        # model_save_path = "models/kcenter_lenet_5.h5"
        model = Lenet5()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        dropout_model = Lenet5_dropout()
        dropout_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        MNIST_al(model, steps, windows, epochs, threshold, results_path, metric, model_save_path, dropout_model)

    # python -u active_learning.py -metric 4 -results results/al_results_lc.csv -model models/lc_lenet_5.h5
    # Cifar-10
    if mode == 1:
        depth = 20
        input_shape = (32, 32, 3)
        steps = 60
        windows = 1000
        epochs = 200
        ###############################################################################
        # resnet20
        threshold = 0.8953
        model = resnet_v1(input_shape=input_shape, depth=depth)
        dropout_model = resnet_v1_dropout(input_shape=input_shape, depth=depth)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        dropout_model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        Cifar10_al(model, steps, windows, epochs, threshold, results_path, metric, model_save_path, dropout_model)
        ###############################################################################
        # NiN
        # threshold = 86.43
        # model = NIN(input_shape, 10)
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=Adam(lr=1e-3),
        #               metrics=['accuracy'])
        # dropout_model = NIN_all(input_shape, 10)
        # dropout_model.compile(loss='categorical_crossentropy',
        #               optimizer=Adam(lr=1e-3),
        #               metrics=['accuracy'])
        #
        # Cifar10_al(model, steps, windows, epochs, threshold, results_path, metric, model_save_path, dropout_model)
        ###############################################################################








