import pathlib
import tensorflow as tf
import numpy as np
from keras.datasets import mnist, cifar10
import argparse
import csv
import keras
from keras.datasets import mnist, cifar10, imdb
from keras.models import load_model
import pathlib
import keras
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.preprocessing.text import Tokenizer
import os
import sys



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


def run_tflite_model(tflite_file, test_image_indices, test_images):

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))

  input_details = interpreter.get_input_details()
  interpreter.resize_tensor_input(
    input_details[0]['index'], (1, 100))
  interpreter.allocate_tensors()
  # interpreter.resizeInputShape(1, 200)

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    if i % 1000 == 0:
      print(i)
    test_image = test_images[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8 or input_details['dtype'] == np.float16:
      input_scale, input_zero_point = input_details["quantization"]
      # print(input_scale)
      test_image = test_image / input_scale + input_zero_point
      # print("haha")
      # print(test_image)
      # return

    # test_image = test_image.reshape(1, 200)
    # print(test_image.shape)
    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    # print(test_image.shape)
    # input_details = interpreter.get_input_details()
    # print(input_details)
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    # print(output.argmax())
    predictions[i] = output.argmax()

  return predictions


def evaluate_model(tflite_file, model_type, model_name, save_path, test_images, test_labels):

  test_image_indices = range(test_images.shape[0])
  # test_image_indices = range(100)

  predictions = run_tflite_model(tflite_file, test_image_indices, test_images)
  # print(predictions)
  # test_labels = test_labels[:100]
  len_label = len(test_labels)
  test_labels = test_labels.reshape(len_label,)
  # print(predictions)
  # print(test_labels)
  # print(test_labels[0])
  accuracy = (np.sum(test_labels == predictions) * 100) / len(test_images)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(test_images)))
  csv_file = open(save_path, "a")
  try:
    writer = csv.writer(csv_file)
    writer.writerow([model_name, accuracy])

  finally:
    csv_file.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", "-model_path",
                      type=str,
                      )
  parser.add_argument("--save_path", "-save_path",
                      type=str,
                      )
  parser.add_argument("--data_type", "-data_type",
                      type=str,
                      default='lenet'
                      )

  args = parser.parse_args()
  model_path = args.model_path
  results_path = args.save_path
  data_type = args.data_type
  # folder_path = "../new_models/RQ1/Yahoo/"
  # save_folder = "../new_models/RQ3/Yahoo/tf8bit/"
  # Image
  if data_type == 'lenet' or data_type == 'vgg' or data_type == 'nin':
    csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
                 "NC", "MCP", "adversarial_al", "KMNC", "DeepGini"]

  if data_type == 'imdb':
    csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "entropy_dropout", "EGL",
                 "NC", "KMNC"]

  if data_type == 'yahoo' or data_type == 'qc':
    csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
                 "NC", "MCP", "KMNC", "DeepGini"]


  tflite_models_dir = pathlib.Path(model_path)
  tflite_models_dir.mkdir(exist_ok=True, parents=True)


  if data_type == 'vgg' or data_type == 'nin':
    (train_images, y_train), (test_images, y_test) = cifar10.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.

    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    if data_type == 'nin':
      x_train_mean = np.mean(train_images, axis=0)
      train_images -= x_train_mean
      test_images -= x_train_mean

  if data_type == 'lenet':
    (train_images, y_train), (test_images, y_test) = mnist.load_data()
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

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

  if data_type == 'qc':
    data, label = get_QC_data("../../data/train_data_pytorch.csv", "../../data/test_data_pytorch.csv")
    train_indices = np.load("../../QC/data/training_indices.npy")
    test_indices = np.load("../../QC/data/test_indices.npy")
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

  for csv_name in csv_names:
    final_path = csv_name + '_gru.tflite'
    tflite_model_quant_file = tflite_models_dir/final_path
    evaluate_model(tflite_model_quant_file, "Quantized", final_path, results_path , test_images, y_test)

  for csv_name in csv_names:
    final_path = csv_name + '_gru_2.tflite'
    tflite_model_quant_file = tflite_models_dir/final_path
    evaluate_model(tflite_model_quant_file, "Quantized", final_path, results_path, test_images, y_test)

  for csv_name in csv_names:
    final_path = csv_name + '_gru_3.tflite'
    tflite_model_quant_file = tflite_models_dir/final_path
    evaluate_model(tflite_model_quant_file, "Quantized", final_path, results_path, test_images, y_test)
