import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

from keras.datasets import cifar10
from keras.utils import np_utils

import tensorly.decomposition as td
import pickle


def decompose_tensor(X, R3, R4):
    core, factors = td.partial_tucker(X.numpy(), modes=[2, 3], rank=[R3, R4])
    return core, factors


def get_compressed_model(model, k=1):
    # iterate over the layers of the model:
    ret_model = Sequential()
    for i, layer in enumerate(model.layers):
        if (isinstance(layer, Conv2D)) and i > 3:
            X = layer.weights[0]
            n_out = X.shape[-1]
            b = layer.weights[1].numpy()
            # print(X.shape)
            r3 = (X.shape[2] * k) // 8
            r4 = (X.shape[3] * k) // 8
            # print(r3, r4)
            # I, core, O = decompose_tensor(X, r3, r4)
            core, factors = decompose_tensor(X, r3, r4)

            I = factors[0]
            I = np.expand_dims(I, axis=0)
            I = np.expand_dims(I, axis=0)

            O = factors[1].T
            O = np.expand_dims(O, axis=0)
            O = np.expand_dims(O, axis=0)

            I_layer = Conv2D(filters=r3, kernel_size=1, strides=(1, 1), padding='valid', activation='relu', use_bias=False, name='test' + str(i))
            core_layer = Conv2D(filters=r4, kernel_size=layer.get_config()['kernel_size'], activation='relu', strides=(1, 1), padding='same', use_bias=False, name='test1' + str(i))
            O_layer = Conv2D(filters=n_out, kernel_size=1, strides=(1, 1), padding='valid', activation='relu', use_bias=True, name='test2' + str(i))
            #
            # I_layer.build(input_shape=[None, None, I.shape[0]])
            # core_layer.build(input_shape=[None, None, core.shape[-2]])
            # O_layer.build(input_shape=[None, None, core.shape[-1]])

            ret_model.add(I_layer)
            ret_model.add(core_layer)
            ret_model.add(O_layer)
            # print("###############")
            # print(I.shape)
            # print("###############")

            I_layer.set_weights([I])
            core_layer.set_weights([core])
            O_layer.set_weights([O, b])
        else:
            ret_model.add(layer)
    return ret_model


def expt_full_model(model):
    losses = []
    accs = []
    opt = keras.optimizers.SGD(lr=0.1, momentum=0.9)
    for k in range(1, 8):
        comp_model = get_compressed_model(model, k)
        comp_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        l, a = comp_model.evaluate(X_test, y=Y_test)
        losses.append(l)
        accs.append(a)
    print(losses)
    print(accs)
    return losses, accs


def expt_reconstruction(model, rank_idx):
    l2err_k = []
    idx = 0
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D) and i > 0:
            idx = i
            break
    X = model.layers[idx].weights[0]
    for k in range(1, 65):
        r3 = X.shape[2]
        r4 = X.shape[3]
        if rank_idx == 2:
            r3 = k
        elif rank_idx == 3:
            r4 = k

        core, factors = decompose_tensor(X, r3, r4)
        I = factors[0]
        O = factors[1].T
        W_hat = np.transpose(np.dot(np.dot(I, core), O), (1, 2, 0, 3))
        print(r3, r4)
        l2err_k.append((np.sum((W_hat - X) ** 2)))
    print(l2err_k)
    return np.array(l2err_k)


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
X_train /= 255
X_test /= 255

# x_train_mean = np.mean(X_train, axis=0)
# X_train -= x_train_mean
# X_test -= x_train_mean
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# batch_size = 100
# lr = 0.001

# model = get_model()
# opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

# weights_path = 'pretrained_cifar10.h5'
# model.load_weights(weights_path)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model = keras.models.load_model("/Users/qiang.hu/PycharmProjects/al_leak/VGG/VGGmodels/VGG16_new.h5")

# model.evaluate(X_test, y=Y_test)

# with open('reconstr_err_r2fixed.pkl', 'wb+') as f:
#   pickle.dump(expt_reconstruction(model, 2), f)
expt_full_model(model)
# with open('reconstr_err_r2fixed.pkl', 'wb+') as f:
#     pickle.dump(expt_full_model(model), f)
