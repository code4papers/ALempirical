import tensorflow as tf
from keras.datasets import mnist, cifar10, imdb
import numpy as np
from keras.models import load_model
import pathlib
import keras
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.preprocessing.text import Tokenizer
import os
import sys


data_type = 'yahoo'
folder_path = "../new_models/RQ1/Yahoo/"
save_folder = "../new_models/RQ3/Yahoo/tf8bit/"
# Image
if data_type == 'lenet' or data_type == 'vgg' or data_type == 'nin':
    csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
                 "NC", "MCP", "adversarial_al", "KMNC", "DeepGini"]

if data_type == 'imdb':
    csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "entropy_dropout",  "EGL",
                 "NC", "KMNC"]

if data_type == 'yahoo' or data_type == 'qc':
    csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
                 "NC", "MCP", "KMNC", "DeepGini"]

def read_yahoo_files():
    text_data_dir = '../Yahoo/data/yahoo_10'
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

# IMDB
# csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "entropy_dropout",  "EGL",
#              "NC", "KMNC"]

# Yahoo, QC
# csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
#              "NC", "MCP", "KMNC", "DeepGini"]

# Cifar10
if data_type == 'vgg' or data_type == 'nin':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.

    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    if data_type == 'nin':
        x_train_mean = np.mean(train_images, axis=0)
        train_images -= x_train_mean
        test_images -= x_train_mean

if data_type == 'lenet':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

if data_type == 'imdb':
    max_features = 20000
    maxlen = 200
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
        num_words=max_features
    )
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    train_images = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    test_images = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)


if data_type == 'qc':
    data, label = get_QC_data("../data/train_data_pytorch.csv", "../data/test_data_pytorch.csv")
    train_indices = np.load("../QC/data/training_indices.npy")
    test_indices = np.load("../QC/data/test_indices.npy")
    train_images = data[train_indices]
    y_train = label[train_indices]
    test_images = data[test_indices]
    y_test = label[test_indices]


if data_type == 'yahoo':
    data, labels, texts = get_Yahoo_data()
    train_index = np.load("../Yahoo/data/train_indices.npy")
    test_index = np.load("../Yahoo/data/test_indices.npy")
    train_images = data[train_index]
    y_train = labels[train_index]
    test_images = data[test_index]
    y_test = labels[test_index]

print(train_images.shape)


# train_images = train_images.reshape(-1, 28, 28, 1)
# test_images = test_images.reshape(-1, 28, 28, 1)
# x_train_mean = np.mean(train_images, axis=0)
# train_images -= x_train_mean
# test_images -= x_train_mean

# IMDB

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(10):
    yield [input_value]


for csv_name in csv_names:
    # saved_model_dir = "../IMDB/IMDB_models/imdb_lstm_glove.h5"
    saved_model_dir = folder_path + csv_name + '_lstm.h5'
    keras_model = load_model(saved_model_dir)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.target_spec.supported_types = [tf.float16]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()

    tflite_models_dir = pathlib.Path(save_folder)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    final_path = csv_name + '_lstm.tflite'
    tflite_model_quant_file = tflite_models_dir/final_path
    tflite_model_quant_file.write_bytes(tflite_model_quant)

