'''
LeNet-5
'''

# usage: python MNISTModel3.py - train the model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.regularizers import l2
from keras.layers import Conv2D, Dense, Input, add, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers, regularizers
from keras.models import Sequential, Model, load_model
from keras.initializers import RandomNormal

from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten,MaxPool2D, Dropout

from keras.models import Model
from keras.datasets import cifar10
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint


import argparse


def lenet5(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = Conv2D(filters=6, kernel_size=5, padding='valid', activation='relu', kernel_initializer='he_normal',
                         kernel_regularizer=l2(0.0001), strides=1)(inp)
    x = MaxPool2D(pool_size=2, strides=1)(x)
    x = Conv2D(filters=16, kernel_size=5, padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001), strides=1)(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = Dense(84, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = Dense(num_classes, name='before_softmax',kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = Activation('softmax', name='predictions')(x)
    # x = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)

    return Model(inp, x, name='lenet-none')


def lenet5_all(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = Conv2D(filters=6, kernel_size=5, padding='valid', activation='relu', kernel_initializer='he_normal',
                         kernel_regularizer=l2(0.0001), strides=1)(inp)
    x = Dropout(0.5)(x, training=True)
    x = MaxPool2D(pool_size=2, strides=1)(x)
    x = Conv2D(filters=16, kernel_size=5, padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001), strides=1)(x)
    x = Dropout(0.5)(x, training=True)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(84, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(num_classes, activation='softmax')(x)

    return Model(inp, x, name='lenet-all')


def NIN(input_shape, num_classes):
    img = Input(shape=input_shape)
    weight_decay = 1e-6
    def NiNBlock(kernel, mlps, strides):
        def inner(x):
            l = Conv2D(mlps[0], kernel, padding='same', strides=strides, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=RandomNormal(stddev = 0.01))(x)
            l = BatchNormalization()(l)
            l = Activation('relu')(l)
            for size in mlps[1:]:
                l = Conv2D(size, 1, padding='same', strides=[1,1], kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=RandomNormal(stddev = 0.05))(l)
                l = BatchNormalization()(l)
                l = Activation('relu')(l)
            return l
        return inner
    l1 = NiNBlock(5, [192, 160, 96], [1,1])(img)
    l1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding = 'same')(l1)
    l1 = Dropout(0.5)(l1)

    l2 = NiNBlock(5, [192, 192, 192], [1,1])(l1)
    l2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding = 'same')(l2)
    l2 = Dropout(0.5)(l2)

    l3 = NiNBlock(3, [192, 192, 10], [1,1])(l2)

    l4 = GlobalAveragePooling2D()(l3)
    l4 = Activation('softmax')(l4)

    model = Model(inputs=img, outputs=l4, name='NIN-none')
    return model


def NIN_all(input_shape, num_classes):
    img = Input(shape=input_shape)
    weight_decay = 1e-6
    def NiNBlock(kernel, mlps, strides):
        def inner(x):
            l = Conv2D(mlps[0], kernel, padding='same', strides=strides, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=RandomNormal(stddev = 0.01))(x)
            l = BatchNormalization()(l)
            l = Activation('relu')(l)
            for size in mlps[1:]:
                l = Conv2D(size, 1, padding='same', strides=[1,1], kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=RandomNormal(stddev = 0.05))(l)
                l = BatchNormalization()(l)
                l = Activation('relu')(l)
            return l
        return inner
    l1 = NiNBlock(5, [192, 160, 96], [1,1])(img)
    l1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding = 'same')(l1)
    l1 = Dropout(0.5)(l1, training=True)

    l2 = NiNBlock(5, [192, 192, 192], [1,1])(l1)
    l2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding = 'same')(l2)
    l2 = Dropout(0.5)(l2, training=True)

    l3 = NiNBlock(3, [192, 192, 10], [1,1])(l2)

    l4 = GlobalAveragePooling2D()(l3)
    l4 = Activation('softmax')(l4)

    model = Model(inputs=img, outputs=l4, name='NIN-all')
    return model


def lenet1(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = Conv2D(filters=4, kernel_size=5, padding='valid', activation='relu', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.0001))(inp)
    x = MaxPool2D(pool_size=2, name='block1_pool1')(x)
    x = Conv2D(filters=12, kernel_size=5, padding='valid', activation='relu', kernel_initializer='he_normal', \
               kernel_regularizer=l2(0.0001))(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)

    return Model(inp, x, name='lenet1-none')


def lenet1_all(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = Conv2D(filters=4, kernel_size=5, padding='valid', activation='relu', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.0001))(inp)
    x = Dropout(0.5)(x, training=True)
    x = MaxPool2D(pool_size=2, name='block1_pool1')(x)
    x = Conv2D(filters=12, kernel_size=5, padding='valid', activation='relu', kernel_initializer='he_normal', \
               kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.5)(x, training=True)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(x)

    return Model(inp, x, name='lenet1-all')

def resnet20(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    weight_decay = 1e-6
    stack_n = 3
    def residual_block(intput, out_channel, increase=False):
        if increase:
            stride = (2, 2)
        else:
            stride = (1, 1)

        pre_bn = BatchNormalization()(intput)
        pre_relu = Activation('relu')(pre_bn)

        conv_1 = Conv2D(out_channel, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(pre_relu)
        bn_1 = BatchNormalization()(conv_1)
        relu1 = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_channel, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(relu1)
        conv_2 = Dropout(0.5)(conv_2)
        if increase:
            projection = Conv2D(out_channel,
                                kernel_size=(1, 1),
                                strides=(2, 2),
                                padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(intput)
            block = add([conv_2, projection])
        else:
            block = add([intput, conv_2])
        return block

        # build model
        # total layers = stack_n * 3 * 2 + 2
        # stack_n = 5 by default, total layers = 32, which is resnet32
        # input: 32x32x3 output: 32x32x16

    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(num_classes, name='before_softmax',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('softmax')(x)
    model = Model(input_tensor, x, name='res20-none')
    return model

def resnet20_all(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    weight_decay = 1e-6
    stack_n = 3
    def residual_block(intput, out_channel, increase=False):
        if increase:
            stride = (2, 2)
        else:
            stride = (1, 1)

        pre_bn = BatchNormalization()(intput)
        pre_relu = Activation('relu')(pre_bn)

        conv_1 = Conv2D(out_channel, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(pre_relu)
#        conv_1 = Dropout(0.5)(conv_1, training=True)
        bn_1 = BatchNormalization()(conv_1)
        relu1 = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_channel, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(relu1)
        conv_2 = Dropout(0.5)(conv_2, training=True)
        if increase:
            projection = Conv2D(out_channel,
                                kernel_size=(1, 1),
                                strides=(2, 2),
                                padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(intput)
#            projection = Dropout(0.5)(projection, training=True)
            block = add([conv_2, projection])
        else:
            block = add([intput, conv_2])
        return block

        # build model
        # total layers = stack_n * 3 * 2 + 2
        # stack_n = 5 by default, total layers = 32, which is resnet32
        # input: 32x32x3 output: 32x32x16

    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
#    x = Dropout(0.5)(x, training=True)
    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(num_classes, name='before_softmax',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('softmax')(x)
    model = Model(input_tensor, x, name='res20-all')
    return model


def build_model(weights_path, input_tensor = None, type='lenet5'):
    if type == 'lenet1':
        if input_tensor is None:
            input_shape = (28, 28, 1)
        model1 = lenet1(input_shape, 10)
        model2 = lenet1_all(input_shape, 10)
        model1.load_weights(weights_path)
        model2.load_weights(weights_path)
        return model1, model2

    elif type == 'lenet5':
        if input_tensor is None:
            input_shape = (28, 28, 1)

        model1 = lenet5(input_shape, 10)
        model2 = lenet5_all(input_shape, 10)
        model1.load_weights(weights_path)
        model2.load_weights(weights_path)
        return model1, model2

    elif  type == 'resnet20':
        if input_tensor is None:
            input_shape = (32, 32, 3)

        model1 = resnet20(input_shape, 10)
        model2 = resnet20_all(input_shape, 10)
        model1.load_weights(weights_path)
        model2.load_weights(weights_path)
        return model1, model2

    elif  type == 'nin':
        if input_tensor is None:
            input_shape = (32, 32, 3)

        model1 = NIN(input_shape, 10)
        model2 = NIN_all(input_shape, 10)
        model1.load_weights(weights_path)
        model2.load_weights(weights_path)
        return model1, model2

    else:
        assert (False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num", "-num",
                        type=int,
                        )
    args = parser.parse_args()
    num = args.num
    input_shape = (32, 32, 3)
    model = NIN(input_shape, 10)
    model.summary()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    model_filename = "NiNmodels/NiN.h5"
    # checkpoint = ModelCheckpoint(model_filename,
    #                              monitor='val_accuracy',
    #                              verbose=1,
    #                              save_best_only=True,
    #                              mode='max',
    #                              period=1)
    # cbks = [checkpoint]
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train,
                         batch_size=128,
                         epochs=200,
                         validation_data=(x_test, y_test),
                         verbose=1)
    model.save("../../new_models/RQ1/NiN/NiN_" + str(num) + ".h5")
    # print(history.history['val_accuracy'][-1])
