# usage: python MNISTLeNet_5.py - train the model

from __future__ import print_function
from keras.utils import to_categorical
import numpy as np
from strategy import *
from Lenet_5 import *
from keras.datasets import mnist, cifar10, cifar100
from resnet20 import *
from NiN.NiNmodel import *
from VGG.VGG16models import *
import tensorflow as tf

from Lenet1.Lenet1_model import *
# from DenseNet.DenseNet import *
from WideResNet.WideResNet import *
from VGG19.VGG19models import *
from VGG19.utils import *
from IMDB.IMDB_model import *
from QC.QC_model import *
from Yahoo.Yahoo_model import *
import csv
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
from art.data_generators import KerasDataGenerator
from art.defences.trainer import AdversarialTrainer


def MNIST_al(model, steps, windows, epochs, results_path, strategy, model_save, dropout_model=None):
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

        elif strategy == 10:
            selected_data, selected_label, remain_data, remain_label = EGL_selection(model, target_data, y_train, windows)
            target_data = remain_data
            y_train = remain_label

        elif metric == 11:
            selected_data, selected_label, remain_data, remain_label = random_selection(model, target_data, y_train,
                                                                                         windows)
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

        # classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
        # pgd = ProjectedGradientDescent(classifier, eps=0.2 / 255, eps_step=0.02 / 255, max_iter=10, num_random_init=1)
        # datagen = ImageDataGenerator(
        #     horizontal_flip=False
        # )
        #
        # art_datagen = KerasDataGenerator(
        #     datagen.flow(x=training_data, y=training_label, batch_size=256, shuffle=True),
        #     size=x_train.shape[0],
        #     batch_size=256,
        # )

        train_len = len(training_data)
        # print("training data len: {}".format(train_len))
        # adv_trainer = AdversarialTrainer(classifier, attacks=pgd, ratio=1.0)
        # adv_trainer.fit_generator(art_datagen, nb_epochs=epochs)
        his = model.fit(training_data, training_label, batch_size=256, shuffle=True, epochs=epochs,
                        validation_data=(x_test, y_test), verbose=1)
        val_acc = his.history['val_accuracy'][-1]
        model.save(model_save)
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([train_len, val_acc])

        finally:
            csv_file.close()
    # np.save("../../data/selected_data/Lenet5/" + str(strategy) + "_x_selected.npy", training_data)
    # np.save("../../data/selected_data/Lenet5/" + str(strategy) + "_y_selected.npy", training_label)
    # np.save("../../data/selected_data/Lenet5/" + str(strategy) + "_x_remain.npy", target_data)
    # np.save("../../data/selected_data/Lenet5/" + str(strategy) + "_y_remain.npy", y_train)



def Cifar10_al(model, steps,  windows, epochs, results_path, metric, model_save_path, dropout_model=None, data_aug=True):
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

        elif metric == 2:
            selected_data, selected_label, remain_data, remain_label = random_selection(model, target_data, y_train,
                                                                                         windows)
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

        elif metric == 10:
            selected_data, selected_label, remain_data, remain_label = EGL_selection(model, target_data, y_train, windows)
            target_data = remain_data
            y_train = remain_label

        elif metric == 11:
            selected_data, selected_label, remain_data, remain_label = random_selection(model, target_data, y_train,
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

        # lr_scheduler = LearningRateScheduler(lr_schedule)

        # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
        #                                cooldown=0,
        #                                patience=5,
        #                                min_lr=0.5e-6)
        # early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
        # callbacks = [lr_reducer, lr_scheduler, early_stop]
        # if data_aug:
        #     datagen.fit(training_data)
        #     # his = model.fit(training_data, training_label,
        #     #       batch_size=128,
        #     #       epochs=epochs,
        #     #       validation_data=(x_test, y_test),
        #     #       shuffle=True,
        #     #       callbacks=callbacks)
        #     his = model.fit_generator(datagen.flow(training_data, training_label, batch_size=128),
        #                               validation_data=(x_test, y_test),
        #                               epochs=epochs, verbose=0, workers=4,
        #                               callbacks=callbacks)
        # else:
        #     early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
        #     # checkpoint = ModelCheckpoint(model_save_path,
        #     #                              monitor='val_accuracy',
        #     #                              verbose=1,
        #     #                              save_best_only=True,
        #     #                              mode='max',
        #     #                              period=1)
        #     cbks = [early_stop]
        #     his = model.fit(x_train, y_train,
        #                     batch_size=128,
        #                     epochs=200,
        #                     callbacks=cbks,
        #                     validation_data=(x_test, y_test),
        #                     verbose=1)
        his = model.fit(training_data, training_label,
                        batch_size=128,
                        epochs=200,
                        validation_data=(x_test, y_test),
                        verbose=1)
        val_acc = his.history['val_accuracy'][-1]
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([train_len, val_acc])

        finally:
            csv_file.close()
    model.save(model_save_path)


def al_VGG16(model, steps,  windows, epochs, results_path, metric, model_save_path, dropout_model=None, data_aug=True):
    print("####################")
    print(tf.__version__)
    print("####################")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalize data.
    target_data = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    idxs_lb = np.zeros(len(target_data), dtype=bool)

    for _ in range(steps):
        if _ != 0:
            model = load_model(model_save_path)
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

        elif metric == 10:
            selected_data, selected_label, remain_data, remain_label = EGL_selection(model, target_data, y_train, windows)
            target_data = remain_data
            y_train = remain_label

        elif metric == 11:
            selected_data, selected_label, remain_data, remain_label = random_selection(model, target_data, y_train,
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

        bestmodelname = model_save_path
        checkPoint = ModelCheckpoint(bestmodelname, monitor="val_accuracy", save_best_only=True, verbose=1)

        def lr_scheduler(epoch):
            initial_lrate = 1e-2
            drop = 0.9
            epochs_drop = 50.0
            lrate = initial_lrate * np.power(drop,
                                             np.floor((1 + epoch) / epochs_drop))
            return lrate

        reduce_lr = callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
        train_datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        train_datagen.fit(training_data)
        train_generator = train_datagen.flow(training_data, training_label, batch_size=128)
        nb_train_samples = training_data.shape[0] // 128
        his = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples,
            epochs=200,
            validation_data=(x_test, y_test),
            validation_steps=10000 // 128,
            callbacks=[checkPoint, reduce_lr])

        val_acc = np.max(his.history['val_accuracy'])
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([train_len, val_acc])

        finally:
            csv_file.close()
        # if train_len >= 25000:
        #     break


def active_learning_run(model_type, results_path, metric, model_save_path):
    if model_type == 'lenet5':
        steps = 20
        windows = 500
        epochs = 5
        model = Lenet5()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        dropout_model = Lenet5_dropout()
        dropout_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        MNIST_al(model, steps, windows, epochs, results_path, metric, model_save_path, dropout_model)
    elif model_type == 'lenet1':
        steps = 20
        windows = 500
        epochs = 5
        model = Lenet1()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        dropout_model = Lenet1_dropout()
        dropout_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        MNIST_al(model, steps, windows, epochs, results_path, metric, model_save_path, dropout_model)
    elif model_type == 'NiN':
        input_shape = (32, 32, 3)
        steps = 10
        windows = 2500
        epochs = 200
        # model = NIN(input_shape, 10)
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=Adam(lr=1e-3),
        #               metrics=['accuracy'])
        model = load_model("/mnt/irisgpfs/users/qihu/pv_env/al_leak/new_models/RQ1/NiN/init.h5")
        dropout_model = NIN_all(input_shape, 10)
        dropout_model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=1e-3),
                      metrics=['accuracy'])
        Cifar10_al(model, steps, windows, epochs, results_path, metric, model_save_path, dropout_model)
    elif model_type == 'VGG16':
        steps = 10
        windows = 2500
        epochs = 200
        model = load_model("/mnt/irisgpfs/users/qihu/pv_env/al_leak/VGG/VGGmodels/init_VGG.h5")
        dropout_model = VGG16_clipped_dropout(input_shape=(32, 32, 3), rate=0.4,
                                              nb_classes=10)
        dropout_model.compile(loss='categorical_crossentropy',
                              optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),
                              metrics=['accuracy'])
        al_VGG16(model, steps, windows, epochs, results_path, metric, model_save_path, dropout_model)

    elif model_type == 'IMDB_lstm' or model_type == 'IMDB_gru':
        from IMDB.active_learning_IMDB import IMDB_al
        steps = 25
        windows = 500
        epochs = 5
        if model_type == 'IMDB_lstm':
            model = IMDB_LSTM_glove(emb_train=True)
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            dropout_model = IMDB_LSTM_glove_dropout()
            dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        else:
            model = IMDB_GRU_new()
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            dropout_model = IMDB_GRU_new_dropout()
            dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        IMDB_al(model, steps, windows, epochs, results_path, metric, model_save_path, dropout_model)
    elif model_type == 'QC_lstm' or model_type == 'QC_gru':
        from QC.active_learning_QC import QC_al
        steps = 10
        windows = 1000
        epochs = 10
        if model_type == 'QC_lstm':
            model = QC_LSTM(emb_train=True)
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            dropout_model = QC_LSTM_dropout()
            dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        else:
            model = QC_GRU(emb_train=True)
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            dropout_model = QC_GRU_dropout()
            dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        QC_al(model, steps, windows, epochs, results_path, metric, model_save_path, dropout_model)
    elif model_type == 'Yahoo_lstm' or model_type == 'Yahoo_gru':
        from Yahoo.active_learning_yahoo import Yahoo_al
        steps = 10
        windows = 178
        epochs = 16
        if model_type == 'Yahoo_lstm':
            model = Yahoo_LSTM()
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            dropout_model = Yahoo_LSTM_dropout()
            dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        else:
            model = Yahoo_GRU()
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            dropout_model = Yahoo_GRU_dropout()
            dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        Yahoo_al(model, steps, windows, epochs, results_path, metric, model_save_path, dropout_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", "-metric",
                        type=int,
                        )
    # 0-entropy 1-BALD 3-k_center 6-margin 7-entropy_dropout 8-margin_dropout
    parser.add_argument("--results", "-results",
                        type=str,
                        )
    parser.add_argument("--model", "-model",
                        type=str,
                        )
    parser.add_argument("--model_type", "-model_type",
                        type=str,
                        )
    args = parser.parse_args()
    metric = args.metric
    results_path = args.results
    model_save_path = args.model
    model_type = args.model_type
    active_learning_run(model_type, results_path, metric, model_save_path)



    # if mode == 2:
    #     # WideResnet
    #     # threshold = 0.7
    #     # init = (32, 32, 3)
    #     # steps = 60
    #     # windows = 5000
    #     # epochs = 200
    #     # model = create_wide_residual_network(init, nb_classes=100, N=4, k=12)
    #     # optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0005)
    #     # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    #     # dropout_model = create_dropout_wide_residual_network(init, nb_classes=100, N=4, k=12)
    #     # dropout_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    #     # al_wideresnet(model, steps, windows, epochs, threshold, results_path, metric, model_save_path, dropout_model)
    #     # VGG19
    #     threshold = 0.65
    #     init = (32, 32, 3)
    #     steps = 60
    #     windows = 2500
    #     epochs = 200
    #     model = vgg19()
    #     optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
    #     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    #     dropout_model = vgg19_dropout()
    #     dropout_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    #     al_wideresnet(model, steps, windows, epochs, threshold, results_path, metric, model_save_path, dropout_model)








