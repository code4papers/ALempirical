from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import keras
from utils.pruned_layers import *
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn.metrics as metrics


import numpy as np

weight_decay = 0.0005





def initial_conv(input):
    x = pruned_Conv2D(16, (3, 3), padding='same',
                      kernel_regularizer=l2(weight_decay))(input)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def expand_conv(init, base, k, strides=(1, 1)):
    x = pruned_Conv2D(base * k, (3, 3), padding='same', strides=strides,
                      kernel_regularizer=l2(weight_decay),
                      )(init)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = pruned_Conv2D(base * k, (3, 3), padding='same',
                      kernel_regularizer=l2(weight_decay))(x)

    skip = pruned_Conv2D(base * k, (1, 1), padding='same', strides=strides,
                      kernel_regularizer=l2(weight_decay),
                      )(init)

    m = Add()([x, skip])

    return m


def conv1_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)
    x = pruned_Conv2D(16 * k, (3, 3), padding='same', activation='linear',
                      kernel_regularizer=l2(weight_decay))(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = pruned_Conv2D(16 * k, (3, 3), padding='same', activation='linear',
                      kernel_regularizer=l2(weight_decay))(x)

    m = Add()([init, x])
    return m


def conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = -1

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)
    x = pruned_Conv2D(32 * k, (3, 3), padding='same',activation='linear',
                      kernel_regularizer=l2(weight_decay))(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = pruned_Conv2D(32 * k, (3, 3), padding='same',activation='linear',
                      kernel_regularizer=l2(weight_decay),
                      )(x)

    m = Add()([init, x])
    return m


def conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = -1

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)
    x = pruned_Conv2D(64 * k, (3, 3), padding='same', activation='linear',
                      kernel_regularizer=l2(weight_decay)
                      )(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = pruned_Conv2D(64 * k, (3, 3), padding='same',activation='linear',
                      kernel_regularizer=l2(weight_decay))(x)

    m = Add()([init, x])
    return m


def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1):
    """
    Creates a Wide Residual Network with specified parameters

    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    ip = Input(shape=input_dim)

    x = initial_conv(ip)
    nb_conv = 4

    x = expand_conv(x, 16, k)
    nb_conv += 2

    for i in range(N - 1):
        x = conv1_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 32, k, strides=(2, 2))
    nb_conv += 2

    for i in range(N - 1):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 64, k, strides=(2, 2))
    nb_conv += 2

    for i in range(N - 1):
        x = conv3_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, kernel_regularizer=l2(weight_decay), activation='softmax')(x)

    model = Model(ip, x)

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model


def create_dropout_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1):
    """
    Creates a Wide Residual Network with specified parameters

    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    ip = Input(shape=input_dim)

    x = initial_conv(ip)
    nb_conv = 4

    x = expand_conv(x, 16, k)
    nb_conv += 2

    for i in range(N - 1):
        x = conv1_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 32, k, strides=(2, 2))
    nb_conv += 2

    for i in range(N - 1):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x, training=True)

    x = expand_conv(x, 64, k, strides=(2, 2))
    nb_conv += 2

    for i in range(N - 1):
        x = conv3_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, kernel_regularizer=l2(weight_decay), activation='softmax')(x)

    model = Model(ip, x)

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model


if __name__ == "__main__":
    from keras.utils import plot_model
    from keras.layers import Input
    from keras.models import Model
    from keras.utils import to_categorical
    init = (32, 32, 3)
    batch_size = 128
    nb_epoch = 200
    nb_classes = 100
    wrn_28_10 = create_wide_residual_network(init, nb_classes=100, N=4, k=12)

    wrn_28_10.summary()
    lr = 0.1
    optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0005)


    def lr_scheduler(epoch):
        initial_lrate = lr
        if 60 <= epoch < 120:
            initial_lrate = 0.02
        if 120 <= epoch < 160:
            initial_lrate = 0.004
        if epoch > 160:
            initial_lrate = 0.0008
        return initial_lrate


    # optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
    wrn_28_10.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    # wrn_28_10.save("models/EGL_wideresnet.h5")

    # print("Finished compiling")
    # print("Building model...")
    #
    # (trainX, trainY), (testX, testY) = cifar100.load_data()
    #
    # trainX = trainX.astype('float32')
    # testX = testX.astype('float32')
    #
    # trainX /= 255.
    # testX /= 255.
    #
    # trainX, testX = color_preprocessing(trainX, testX)
    #
    # Y_train = keras.utils.to_categorical(trainY, nb_classes)
    # Y_test = keras.utils.to_categorical(testY, nb_classes)
    #
    # generator = ImageDataGenerator(
    #                                vertical_flip=True,
    #                                horizontal_flip=True
    # )
    #
    # generator.fit(trainX, seed=0)
    #
    # # Load model
    #
    # reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    #
    # early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=20)
    # model_checkpoint = ModelCheckpoint("WideResnet.h5", monitor="val_accuracy",
    #                                    save_best_only=True)
    #
    # callbacks = [reduce_lr, early_stopper, model_checkpoint]
    #
    # wrn_28_10.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
    #                     steps_per_epoch=len(trainX) / batch_size, epochs=nb_epoch,
    #                     callbacks=callbacks,
    #                     validation_data=(testX, Y_test),
    #                     validation_steps=testX.shape[0] // batch_size, verbose=1)
    #
    # yPreds = wrn_28_10.predict(testX)
    # yPred = np.argmax(yPreds, axis=1)
    # yTrue = testY
    #
    # accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    # error = 100 - accuracy
    # print("Accuracy : ", accuracy)
    # print("Error : ", error)
