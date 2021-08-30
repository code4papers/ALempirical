import numpy as np
import tensorflow as tf
from tensorflow_model_optimization import sparsity
from kerassurgeon import identify
from keras.datasets import mnist, cifar10
# from kerassurgeon.operations import delete_channels
from keras.utils import to_categorical
import keras
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.preprocessing.text import Tokenizer
import os
import sys
import csv
import argparse


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

# x_test = np.expand_dims(x_test, 3)


def get_layers_index(model):
    dense_layer_list = []
    conv_layer_list = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            conv_layer_list.append(i)
        if isinstance(layer, keras.layers.Dense):
            dense_layer_list.append(i)
    return dense_layer_list, conv_layer_list


def prune_model(model, percentage, x):
    dense_layer_list, conv_layer_list = get_layers_index(model)
    # conv_layer_list = conv_layer_list[-1:]
    # print(dense_layer_list)
    # print(conv_layer_list)
    for layer_index in dense_layer_list:
        # print(layer_index)
        weights_bias = model.layers[layer_index].get_weights()
        apoz = identify.get_apoz(model, model.layers[layer_index], x)
        pruned_num = int(len(apoz) * percentage)
        if pruned_num > 0:
            selected_index = np.argsort(apoz)[:pruned_num]
            weights_bias[0][selected_index] = 0
            model.layers[layer_index].set_weights(weights_bias)
            del weights_bias
        else:
            del weights_bias
            continue
    for layer_index in conv_layer_list:
        weights_bias = model.layers[layer_index].get_weights()
        apoz = identify.get_apoz(model, model.layers[layer_index], x)
        print(apoz)
        pruned_num = int(len(apoz) * percentage)
        if pruned_num > 0:
            selected_index = np.argsort(apoz)[:pruned_num]
            weights_bias[0][:, :, :, selected_index] = 0
            model.layers[layer_index].set_weights(weights_bias)
            del weights_bias
        else:
            del weights_bias
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        default="../../new_models/RQ1/Yahoo/"
                        )
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        default="../../new_results/RQ3/Yahoo/pruning/apoz_gru_1.csv"
                        )
    parser.add_argument("--data_type", "-data_type",
                        type=str,
                        default='yahoo'
                        )

    args = parser.parse_args()
    model_path = args.model_path
    results_path = args.save_path
    data_type = args.data_type

    # data_type = 'nin'
    folder_path = model_path
    save_folder = results_path
    # Image
    if data_type == 'lenet' or data_type == 'vgg' or data_type == 'nin':
        # csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
        #              "NC", "MCP", "adversarial_al", "KMNC", "DeepGini"]
        csv_names = ["random"]

        # csv_names = ["entropy_dropout", "margin_dropout", "EGL",
        #              "NC", "MCP", "adversarial_al", "KMNC", "DeepGini"]

    if data_type == 'imdb':
        # csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "entropy_dropout", "EGL",
        #              "NC", "KMNC"]
        csv_names = ["random"]
        # csv_names = ["k_center", "entropy_dropout", "EGL",
        #              "NC", "KMNC"]

    if data_type == 'yahoo' or data_type == 'qc':
        # csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
        #              "NC", "MCP", "KMNC", "DeepGini"]
        csv_names = ["random"]

    if data_type == 'vgg' or data_type == 'nin':
        (train_images, y_train), (test_images, y_test) = cifar10.load_data()

        # Normalize the input image so that each pixel value is between 0 to 1.

        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0
        if data_type == 'nin':
            x_train_mean = np.mean(train_images, axis=0)
            train_images -= x_train_mean
            test_images -= x_train_mean
        y_test = to_categorical(y_test, 10)

    if data_type == 'lenet':
        (train_images, y_train), (test_images, y_test) = mnist.load_data()
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0
        train_images = train_images.reshape(-1, 28, 28, 1)
        test_images = test_images.reshape(-1, 28, 28, 1)
        y_test = to_categorical(y_test, 10)

    if data_type == 'imdb':
        max_features = 20000
        maxlen = 200
        (x_train, y_train), (x_val, y_test) = keras.datasets.imdb.load_data(
            num_words=max_features
        )
        print(len(x_train), "Training sequences")
        print(len(x_val), "Validation sequences")
        train_images = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
        test_images = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
        y_test = to_categorical(y_test, 2)

    if data_type == 'qc':
        data, label = get_QC_data("../../data/train_data_pytorch.csv", "../../data/test_data_pytorch.csv")
        train_indices = np.load("../../QC/data/training_indices.npy")
        test_indices = np.load("../../QC/data/test_indices.npy")
        train_images = data[train_indices]
        y_train = label[train_indices]
        test_images = data[test_indices]
        y_test = label[test_indices]
        y_test = to_categorical(y_test, 7)

    if data_type == 'yahoo':
        data, labels, texts = get_Yahoo_data()
        train_index = np.load("../../Yahoo/data/train_indices.npy")
        test_index = np.load("../../Yahoo/data/test_indices.npy")
        train_images = data[train_index]
        y_train = labels[train_index]
        test_images = data[test_index]
        y_test = labels[test_index]
        y_test = to_categorical(y_test, 10)

    for csv_name in csv_names:
        print(csv_name)
        model_path = folder_path + csv_name + '_gru_1.h5'
        model = keras.models.load_model(model_path)
        # model.summary()
        ori_weights = model.get_weights()
        with open(save_folder, 'a', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow([model_path])
            file.close()
        for percentage in [0.1 * _ for _ in range(10)]:
            # pruned_model = tf.keras.models.clone_model(model)
            prune_model(model, percentage, test_images)
            pruned_test_loss, pruned_test_acc = model.evaluate(test_images, y_test)
            model.set_weights(ori_weights)
            with open(save_folder, 'a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                # writer.writerow(header)
                writer.writerow([percentage, pruned_test_acc])
        del model

    for csv_name in csv_names:
        print(csv_name)
        model_path = folder_path + csv_name + '_gru_2.h5'
        model = keras.models.load_model(model_path)
        ori_weights = model.get_weights()
        with open(save_folder, 'a', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow([model_path])
            file.close()
        for percentage in [0.1 * _ for _ in range(10)]:
            # pruned_model = tf.keras.models.clone_model(model)
            prune_model(model, percentage, test_images)
            pruned_test_loss, pruned_test_acc = model.evaluate(test_images, y_test)
            model.set_weights(ori_weights)
            with open(save_folder, 'a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                # writer.writerow(header)
                writer.writerow([percentage, pruned_test_acc])

    for csv_name in csv_names:
        print(csv_name)
        model_path = folder_path + csv_name + '_gru_3.h5'
        model = keras.models.load_model(model_path)
        ori_weights = model.get_weights()
        with open(save_folder, 'a', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow([model_path])
            file.close()
        for percentage in [0.1 * _ for _ in range(10)]:
            # pruned_model = tf.keras.models.clone_model(model)
            prune_model(model, percentage, test_images)
            pruned_test_loss, pruned_test_acc = model.evaluate(test_images, y_test)
            model.set_weights(ori_weights)
            with open(save_folder, 'a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                # writer.writerow(header)
                writer.writerow([percentage, pruned_test_acc])
            # del pruned_model
    model_name = 'gru'
    if data_type == 'nin':
        model_name = "NiN"
    if data_type == 'vgg':
        model_name = 'VGG16'

    for i in range(1, 4):
        model_path = folder_path + model_name + '_' + str(i) + '.h5'
        model = keras.models.load_model(model_path)
        ori_weights = model.get_weights()
        with open(save_folder, 'a', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow([model_path])
            file.close()
        for percentage in [0.1 * _ for _ in range(10)]:
            # pruned_model = tf.keras.models.clone_model(model)
            prune_model(model, percentage, test_images)
            pruned_test_loss, pruned_test_acc = model.evaluate(test_images, y_test)
            model.set_weights(ori_weights)
            with open(save_folder, 'a', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                # writer.writerow(header)
                writer.writerow([percentage, pruned_test_acc])




