from warnings import filterwarnings

filterwarnings("ignore")

# Data loading and pre-processing
from keras.datasets import cifar10, mnist, cifar100
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
from keras import backend as K
import tensorflow as tf
import keras
import glob
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys


import numpy as np
import os
import csv
from keras.utils import to_categorical
from WideResnet_pruned import *

# Model related imports
from utils.hyperparams import parse_args
from utils.pruned_layers import pruned_Conv2D
from utils.utils import create_dir_if_not_exists
from utils.model import get_model, convert_to_masked_model

def color_preprocessing(x_train, x_test):

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test

def get_QC_data(train_data_path, test_data_path):
    max_features = 10000
    max_len = 100
    texts = []
    labels = []
    qc_train = pd.read_csv(train_data_path,
                     names=['num', 'title', 'description', 'category'])
    # print(qc['description'][0])
    for i in range(len(qc_train)):
        texts.append(str(qc_train['description'][i]))
        labels.append(qc_train['category'][i])

    qc_test = pd.read_csv(test_data_path,
                          names=['num', 'title', 'description', 'category'])
    # print(qc['description'][0])
    for i in range(len(qc_test)):
        texts.append(str(qc_test['description'][i]))
        labels.append(qc_test['category'][i])

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_len)
    labels = np.asarray(labels)
    return data, labels

def read_yahoo_files():
    text_data_dir = '../../Yahoo/data/yahoo_10'
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    texts.append(f.read())
                    f.close()
                    labels.append(label_id)

    # labels = to_categorical(np.asarray(labels))
    return texts, labels, labels_index


def get_Yahoo_data():
    max_features = 20000
    max_len = 1000
    texts, labels, labels_index = read_yahoo_files()
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    labels = np.asarray(labels)
    return data, labels, texts


def pruning_model(model_path, optimizer, result_path, x_test, y_test, model_type):
    with open(result_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow([model_path])
        file.close()
    model = keras.models.load_model(model_path)
    model.summary()
    layers = get_layers_index(model)

    # convert the layers to masked layers
    if model_type == 'wideresnet':
        layers = get_layers_index_wide(model)
        pruned_model = create_wide_residual_network((32, 32, 3), nb_classes=100, N=4, k=12)
        pruned_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        pruned_model.summary()
        # for i, layer in enumerate(pruned_model.layers):
        for i in range(len(pruned_model.layers)):
            if isinstance(pruned_model.layers[i], pruned_Conv2D):
                pruned_model.layers[i].set_weights(model.layers[i].get_weights() + [pruned_model.layers[i].get_mask()])
            else:
                pruned_model.layers[i].set_weights(model.layers[i].get_weights())
        # weights = model.get_weights()
        # pruned_model.set_weights(weights)
    else:
        pruned_model = convert_to_masked_model(model)
        pruned_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # optimizer = SGD(lr=0.1, momentum=0.9)
    # pruned_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # pruned_test_loss, pruned_test_acc = pruned_model.evaluate(x_test, y_test)
    # print("Pruned accuracy: {}".format(pruned_test_acc))
    # pruned_model.summary()
    for pruning_percentage in list(range(0, 100, 10)):
        # Read the weights
        header = [model_path, "^", "^"]
        lines = []
        for layer_index in layers:
            # print(layer_index)
            weights = pruned_model.layers[layer_index].get_weights()[0]
            # Getting the indices of weight values above threshold of pruning_percentage
            index = np.abs(weights) > np.percentile(np.abs(weights), pruning_percentage)
            pruned_model.layers[layer_index].set_mask(index.astype(float))

        pruned_test_loss, pruned_test_acc = pruned_model.evaluate(x_test, y_test)
        lines.append([pruning_percentage, pruned_test_loss, pruned_test_acc])
        pruned_model.layers[layer_index].set_mask(np.ones(pruned_model.layers[layer_index].get_weights()[0].shape))
        with open(result_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            # writer.writerow(header)
            writer.writerows(lines)


def get_layers_index(model):
    layer_list = []
    flag = 0
    for i, layer in enumerate(model.layers):
        if i == 0 and isinstance(layer, keras.layers.InputLayer):
            flag = 1
        if flag == 0:
            if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                layer_list.append(i)
        elif flag == 1:
            if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                layer_list.append(i - 1)
    return layer_list


def get_layers_index_wide(model):
    layer_list = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            layer_list.append(i)
    return layer_list


def get_data(model_type):
    if model_type == 'lenet':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype(np.float32).reshape(-1, 28, 28, 1)
        x_test = x_test.astype(np.float32).reshape(-1, 28, 28, 1)
        x_train /= 255
        x_test /= 255
        n_output = 10
        y_train = np_utils.to_categorical(y_train, n_output)
        y_test = np_utils.to_categorical(y_test, n_output)
    elif model_type == 'nin':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
    elif model_type == 'vgg16':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
    elif model_type == 'wideresnet':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train, x_test = color_preprocessing(x_train, x_test)
        y_train = to_categorical(y_train, 100)
        y_test = to_categorical(y_test, 100)
    elif model_type == 'imdb':
        max_features = 20000
        maxlen = 200
        (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
            num_words=max_features
        )
        x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
        y_train = to_categorical(y_train, 2)
        y_test = to_categorical(y_val, 2)
    elif model_type == 'qc':
        data, label = get_QC_data("../../data/train_data_pytorch.csv", "../../data/test_data_pytorch.csv")
        test_indices = np.load("../../QC/data/test_indices.npy")
        train_indices = np.load("../../QC/data/training_indices.npy")
        x_train = data[train_indices]
        x_test = data[test_indices]
        y_test = label[test_indices]
        y_train = label[train_indices]
        y_train = to_categorical(y_train, 7)
        y_test = to_categorical(y_test, 7)

    elif model_type == 'yahoo':
        data, labels, texts = get_Yahoo_data()
        train_index = np.load("../../Yahoo/data/train_indices.npy")
        test_index = np.load("../../Yahoo/data/test_indices.npy")
        x_train = data[train_index]
        y_train = labels[train_index]
        x_test = data[test_index]
        y_test = labels[test_index]
        # print(len(y_test))

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    # model_path =
    # model_paths = glob.glob("../../Yahoo/models/*")
    model_type = 'qc'
    rnn_type = '_gru'
    folder_path = "../../new_models/RQ1/QC/"
    save_folder = "../../new_results/RQ3/QC/pruning/QC_gru_1.csv"
    # csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
    #              "NC", "MCP", "adversarial_al", "KMNC", "DeepGini", "Lenet5"]
    # Lenet
    if model_type == 'lenet':
        # csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
        #              "NC", "MCP", "adversarial_al", "KMNC", "DeepGini"]
        csv_names = ["random"]
        x_train, y_train, x_test, y_test = get_data(model_type)
        optimizer = "adam"

    # QC
    if model_type == 'qc':
        # csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
        #              "NC", "MCP", "KMNC", "DeepGini"]
        csv_names = ["random"]
        x_train, y_train, x_test, y_test = get_data(model_type)
        optimizer = "adam"

    if model_type == 'yahoo':
        # csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
        #              "NC", "MCP", "KMNC", "DeepGini"]
        csv_names = ["random"]
        x_train, y_train, x_test, y_test = get_data(model_type)
        optimizer = "adam"

    # NiN
    if model_type == 'nin':
        # csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
        #             "NC", "MCP", "adversarial_al", "KMNC", "DeepGini"]
        csv_names = ["random"]
        x_train, y_train, x_test, y_test = get_data(model_type)
        optimizer = Adam(lr=1e-3)

    if model_type == 'vgg16':
        # csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
        #              "NC", "MCP", "adversarial_al", "KMNC", "DeepGini"]
        csv_names = ["random"]
        x_train, y_train, x_test, y_test = get_data(model_type)
        optimizer = SGD(lr=1e-2, momentum=0.9)

    if model_type == 'imdb':
        # csv_names = ["entropy", "BALD", "k_center", "entropy_dropout", "EGL",
        #              "NC", "KMNC", "DSA", "LSA"]
        csv_names = ["random"]
        x_train, y_train, x_test, y_test = get_data(model_type)
        optimizer = "adam"

    # result_path = "results/yahoo_results.csv"
    for csv_name in csv_names:
        model_path = folder_path + csv_name + rnn_type + '_1.h5'
        pruning_model(model_path, optimizer, save_folder, x_test, y_test, model_type)
    # model_path = folder_path + 'Lenet5.h5'
    # pruning_model(model_path, optimizer, save_folder, x_test, y_test, model_type)
