import keras
from keras.datasets import mnist
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K

import numpy as np
import os
import time
import argparse
from sklearn.model_selection import StratifiedShuffleSplit
import json


def Lenet1():
    model = Sequential()
    # block1
    model.add(Conv2D(4, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool1'))
    # block2
    model.add(Conv2D(12, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))

    return model


def Lenet1_dropout():
    model = Sequential()
    # block1
    model.add(Conv2D(4, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool1'))
    # block2
    model.add(Conv2D(12, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))

    return model

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
# y_test = to_categorical(y_test, 10)
# y_train = to_categorical(y_train, 10)
#
# model = Lenet1()
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.summary()
# his = model.fit(x_train, y_train, batch_size=256, shuffle=True, epochs=30,
#                 validation_data=(x_test, y_test), verbose=1)
#
# print(his.history['val_accuracy'])
# model.save("../new_models/RQ1/Lenet1/Lenet1_4.h5")

