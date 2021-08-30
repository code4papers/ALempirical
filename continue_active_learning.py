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
from DenseNet.DenseNet import *
from WideResNet.WideResNet import *
import csv


def al_VGG16(model, steps,  windows, epochs, threshold, results_path, metric, model_save_path, dropout_model=None, data_aug=True):
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
        bestmodelname = model_save_path + str(_) + ".h5"
        if _ != 0:
            model = load_model(model_save_path + str(_ - 1) + ".h5")
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

        # bestmodelname = model_save_path
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
        nb_train_samples = x_train.shape[0]
        his = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // 128,
            epochs=200,
            validation_data=(x_test, y_test),
            validation_steps=10000 // 128,
            callbacks=[checkPoint, reduce_lr])
        np.save("data/remain_cifar10_data.npy", remain_data)
        val_acc = np.max(his.history['val_accuracy'])
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([train_len, val_acc])

        finally:
            csv_file.close()
        # if val_acc > threshold or train_len >= 25000:
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
    mode = 1

    # python -u active_learning.py -metric 4 -results results/al_results_lc.csv -model models/lc_lenet_5.h5
    # Cifar-10
    if mode == 1:
        # depth = 20
        input_shape = (32, 32, 3)
        steps = 50
        windows = 1000
        epochs = 200
        # VGG16
        threshold = 0.9
        # steps = 60
        # windows = 1000
        # epochs = 200
        # model = VGG16_clipped(input_shape=(32, 32, 3), rate=0.4,
        #                       nb_classes=10)
        # vgg16 = VGG16(weights='imagenet', include_top=False)
        # layer_dict = dict([(layer.name, layer) for layer in vgg16.layers])
        # for l in model.layers:
        #     if l.name in layer_dict:
        #         model.get_layer(name=l.name).set_weights(layer_dict[l.name].get_weights())
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),
        #               metrics=['accuracy'])
        model = load_model("/mnt/irisgpfs/users/qihu/pv_env/al_leak/VGG/VGGmodels/init_VGG.h5")
        # dropout_model = VGG16_clipped_dropout(input_shape=(32, 32, 3), rate=0.4,
        #                       nb_classes=10)
        # dropout_model.compile(loss='categorical_crossentropy',
        #               optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),
        #               metrics=['accuracy'])
        al_VGG16(model, steps, windows, epochs, threshold, results_path, metric, model_save_path, dropout_model)
