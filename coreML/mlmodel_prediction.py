import coremltools
# from PIL import Image
import numpy as np
import keras
import glob
from keras.datasets import cifar10, mnist, cifar100
from keras.models import load_model
import glob
import csv
import gc
import os
import sys
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.preprocessing.text import Tokenizer
import keras.backend as K


def color_preprocessing(x_train, x_test):

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


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

# # Cifar10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
# x_train_mean = np.mean(x_train, axis=0)
# x_train -= x_train_mean
# x_test -= x_train_mean
# # model_paths = glob.glob("VGGcoreMLmodels/*")
# folder_path = "../new_models/RQ3/NiN/"
# # csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
# #              "NC", "MCP", "adversarial_al", "KMNC", "DeepGini"]
# csv_names = ["random"]
# save_folder = "../new_results/RQ3/NiN/quantization/"
# bits = [2, 4, 8]
# for bit in bits:
#     for csv_name in csv_names:
#         # for _ in range(20):
#         # # model_path = "NiNcoreMLmodels/EGL_NiN_2bit.mlmodel"
#         #     print(csv_name)
#         #     # model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name +'_1.mlmodel')
#         #     model = coremltools.models.MLModel(folder_path + csv_name + '_' + str(_) + '.mlmodel')
#         #     # print(model)
#         #     # x_train_mean = np.mean(x_train, axis=0)
#         #     # x_test = x_test - x_train_mean
#         #     rightNum = 0
#         #     for i in range(len(x_test)):
#         #         result = model.predict({'block1_conv1_input': x_test[i].reshape(1, 32, 32, 3)})
#         #         # print(result['Identity'][0])
#         #         # print(y_test[i])
#         #         if np.argmax(result['Identity'][0]) == y_test[i]:
#         #             rightNum += 1
#         #         if i % 1000 == 0:
#         #             print("right num: ", rightNum)
#         #         # print("num: %d, right: %d" % (i, rightNum))
#         #
#         #     print("acc")
#         #     print(rightNum/len(x_test))
#         #     csv_file = open(save_folder + csv_name + '.csv', "a")
#         #     try:
#         #         writer = csv.writer(csv_file)
#         #         writer.writerow([csv_name + '_' + str(_), rightNum/len(x_test)])
#         #
#         #     finally:
#         #         csv_file.close()
#         #     K.clear_session()
#         #     del model
#         #     gc.collect()
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_1.mlmodel')
#         print(model)
#         rightNum = 0
#         for i in range(len(x_test)):
#             if i % 1000 == 0:
#                 print("right num: ", rightNum)
#             result = model.predict({'input_1': x_test[i].reshape(1, 32, 32, 3)})
#             if np.argmax(result['Identity'][0]) == y_test[i]:
#                 rightNum += 1
#             # print("num: %d, right: %d" % (i, rightNum))
#
#         print("acc")
#         print(rightNum / len(x_test))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow([csv_name + '_1', rightNum / len(x_test)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()
#         ############################
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_2.mlmodel')
#         rightNum = 0
#         for i in range(len(x_test)):
#             if i % 1000 == 0:
#                 print("right num: ", rightNum)
#             result = model.predict({'input_1': x_test[i].reshape(1, 32, 32, 3)})
#             if np.argmax(result['Identity'][0]) == y_test[i]:
#                 rightNum += 1
#             # print("num: %d, right: %d" % (i, rightNum))
#
#         print("acc")
#         print(rightNum / len(x_test))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow([csv_name + '_2', rightNum / len(x_test)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()
#         #############################
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_3.mlmodel')
#         rightNum = 0
#         for i in range(len(x_test)):
#             if i % 1000 == 0:
#                 print("right num: ", rightNum)
#             result = model.predict({'input_1': x_test[i].reshape(1, 32, 32, 3)})
#             if np.argmax(result['Identity'][0]) == y_test[i]:
#                 rightNum += 1
#             # print("num: %d, right: %d" % (i, rightNum))
#
#         print("acc")
#         print(rightNum / len(x_test))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow([csv_name + '_3', rightNum / len(x_test)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()


# # MNIST
# folder_path = "../new_models/RQ3/Lenet5/"
# save_folder = "../new_results/RQ3/Lenet5/quantization/"
# # csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
# #              "NC", "MCP", "adversarial_al", "KMNC", "DeepGini"]
# csv_names = ["random"]
# bits = [2, 4, 8]
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
# for csv_name in csv_names:
#     for bit in bits:
#         # for _ in range(120):
#         #     model = coremltools.models.MLModel(folder_path + csv_name + '_' + str(_) + '.mlmodel')
#         #     rightNum = 0
#         #     # print(model)
#         #     for i in range(len(x_test)):
#         #         if i % 1000 == 0:
#         #             print("right num: ", rightNum)
#         #         result = model.predict({'input_1': x_test[i].reshape(1, 28, 28, 1)})
#         #         if np.argmax(result['Identity'][0]) == y_test[i]:
#         #             rightNum += 1
#         #         # print("num: %d, right: %d" % (i, rightNum))
#         #
#         #     print("acc")
#         #     print(rightNum / len(x_test))
#         #     csv_file = open(save_folder + csv_name + '.csv', "a")
#         #     try:
#         #         writer = csv.writer(csv_file)
#         #         writer.writerow([csv_name + '_' + str(_), rightNum/len(x_test)])
#         #
#         #     finally:
#         #         csv_file.close()
#         #     K.clear_session()
#         #     del model
#         #     gc.collect()
#         #############################
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_1.mlmodel')
#         # print(model)
#         rightNum = 0
#         for i in range(len(x_test)):
#             if i % 1000 == 0:
#                 print("right num: ", rightNum)
#             result = model.predict({'input_1': x_test[i].reshape(1, 28, 28, 1)})
#             if np.argmax(result['Identity'][0]) == y_test[i]:
#                 rightNum += 1
#             # print("num: %d, right: %d" % (i, rightNum))
#
#         print("acc")
#         print(rightNum / len(x_test))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow([csv_name + '_1', rightNum / len(x_test)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()
#         #############################
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_2.mlmodel')
#         rightNum = 0
#         for i in range(len(x_test)):
#             if i % 1000 == 0:
#                 print("right num: ", rightNum)
#             result = model.predict({'input_1': x_test[i].reshape(1, 28, 28, 1)})
#             if np.argmax(result['Identity'][0]) == y_test[i]:
#                 rightNum += 1
#             # print("num: %d, right: %d" % (i, rightNum))
#
#         print("acc")
#         print(rightNum / len(x_test))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow([csv_name + '_2', rightNum / len(x_test)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()
#         #############################
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_3.mlmodel')
#         rightNum = 0
#         for i in range(len(x_test)):
#             if i % 1000 == 0:
#                 print("right num: ", rightNum)
#             result = model.predict({'input_1': x_test[i].reshape(1, 28, 28, 1)})
#             if np.argmax(result['Identity'][0]) == y_test[i]:
#                 rightNum += 1
#             # print("num: %d, right: %d" % (i, rightNum))
#
#         print("acc")
#         print(rightNum / len(x_test))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow([csv_name + '_3', rightNum / len(x_test)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()


# # # IMDB
# folder_path = "../new_models/RQ4/IMDB_4bit/quantization/"
# save_folder = "../new_results/RQ4/IMDB/quantization_"
# # csv_names = ["entropy", "BALD", "k_center", "entropy_dropout",  "EGL",
# #              "NC", "KMNC", "DSA", "LSA"]
# csv_names = ["NC_lstm"]
#
# # csv_names = ["EGL_1"]
#
#
# max_features = 20000
# maxlen = 200
# (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
#     num_words=max_features
# )
# bits = [4]
# print(len(x_train), "Training sequences")
# print(len(x_val), "Validation sequences")
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
# # print(model_paths)
# # model_paths = ["IMDBcoreMLmodels/entropy_dropout_imdb_lstm_2bit.mlmodel", "IMDBcoreMLmodels/entropy_dropout_imdb_lstm_3bit.mlmodel", "IMDBcoreMLmodels/entropy_dropout_imdb_lstm_4bit.mlmodel"]
# for bit in bits:
#     for csv_name in csv_names:
#         for _ in range(44, 50):
#             model = coremltools.models.MLModel(folder_path + csv_name + '_' + str(_) + '.mlmodel')
#             rightNum = 0
#             for i in range(len(x_val)):
#                 result = model.predict({'input_1': x_val[i].reshape(1, 200)})
#                 if np.argmax(result['Identity'][0]) == y_val[i]:
#                     rightNum += 1
#                 if i % 1000 == 0:
#                     print("num: %d, right: %d" % (i, rightNum))
#             csv_file = open(save_folder + csv_name + '.csv', "a")
#             try:
#                 writer = csv.writer(csv_file)
#                 writer.writerow([csv_name + '_' + str(_), rightNum / len(x_val)])
#             finally:
#                 csv_file.close()
#             K.clear_session()
#             del model
#             gc.collect()


# folder_path = "../new_models/RQ4/IMDB_4bit_2/quantization/"
# save_folder = "../new_results/RQ4/IMDB/quantization_"
# # csv_names = ["entropy", "BALD", "k_center", "entropy_dropout",  "EGL",
# #              "NC", "KMNC", "DSA", "LSA"]
# csv_names = ["NC_lstm"]
#
# # csv_names = ["EGL_1"]
#
#
# max_features = 20000
# maxlen = 200
# (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
#     num_words=max_features
# )
# bits = [4]
# print(len(x_train), "Training sequences")
# print(len(x_val), "Validation sequences")
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
# # print(model_paths)
# # model_paths = ["IMDBcoreMLmodels/entropy_dropout_imdb_lstm_2bit.mlmodel", "IMDBcoreMLmodels/entropy_dropout_imdb_lstm_3bit.mlmodel", "IMDBcoreMLmodels/entropy_dropout_imdb_lstm_4bit.mlmodel"]
# for bit in bits:
#     for csv_name in csv_names:
#         for _ in range(22, 50):
#             model = coremltools.models.MLModel(folder_path + csv_name + '_' + str(_) + '.mlmodel')
#             rightNum = 0
#             for i in range(len(x_val)):
#                 result = model.predict({'input_1': x_val[i].reshape(1, 200)})
#                 if np.argmax(result['Identity'][0]) == y_val[i]:
#                     rightNum += 1
#                 if i % 1000 == 0:
#                     print("num: %d, right: %d" % (i, rightNum))
#             csv_file = open(save_folder + csv_name + '.csv', "a")
#             try:
#                 writer = csv.writer(csv_file)
#                 writer.writerow([csv_name + '_' + str(_), rightNum / len(x_val)])
#             finally:
#                 csv_file.close()
#             K.clear_session()
#             del model
#             gc.collect()
#
#
# folder_path = "../new_models/RQ3/IMDB/"
# save_folder = "../new_results/RQ3/IMDB/quantization/"
# # csv_names = ["entropy", "BALD", "k_center", "entropy_dropout",  "EGL",
# #              "NC", "KMNC", "DSA", "LSA"]
# csv_names = ["random"]
#
# # csv_names = ["EGL_1"]
#
#
# max_features = 20000
# maxlen = 200
# (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
#     num_words=max_features
# )
# bits = [2, 4, 8]
# print(len(x_train), "Training sequences")
# print(len(x_val), "Validation sequences")
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
# # print(model_paths)
# # model_paths = ["IMDBcoreMLmodels/entropy_dropout_imdb_lstm_2bit.mlmodel", "IMDBcoreMLmodels/entropy_dropout_imdb_lstm_3bit.mlmodel", "IMDBcoreMLmodels/entropy_dropout_imdb_lstm_4bit.mlmodel"]
# for bit in bits:
#     for csv_name in csv_names:
#         # for _ in range(50):
#         #     model = coremltools.models.MLModel(folder_path + csv_name + '_' + str(_) + '.mlmodel')
#         #     rightNum = 0
#         #     for i in range(len(x_val)):
#         #         result = model.predict({'input_1': x_val[i].reshape(1, 200)})
#         #         if np.argmax(result['Identity'][0]) == y_val[i]:
#         #             rightNum += 1
#         #         if i % 1000 == 0:
#         #             print("num: %d, right: %d" % (i, rightNum))
#         #     csv_file = open(save_folder + csv_name + '.csv', "a")
#         #     try:
#         #         writer = csv.writer(csv_file)
#         #         writer.writerow([csv_name + '_' + str(_), rightNum / len(x_val)])
#         #     finally:
#         #         csv_file.close()
#         #     K.clear_session()
#         #     del model
#         #     gc.collect()
#
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_lstm_1.mlmodel')
#         rightNum = 0
#         for i in range(len(x_val)):
#             result = model.predict({'input_1': x_val[i].reshape(1, 200)})
#             if np.argmax(result['Identity'][0]) == y_val[i]:
#                 rightNum += 1
#             if i % 1000 == 0:
#                 print("num: %d, right: %d" % (i, rightNum))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow([csv_name + '_lstm_1', rightNum / len(x_val)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()
#
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_lstm_2.mlmodel')
#         rightNum = 0
#         for i in range(len(x_val)):
#             result = model.predict({'input_1': x_val[i].reshape(1, 200)})
#             if np.argmax(result['Identity'][0]) == y_val[i]:
#                 rightNum += 1
#             if i % 1000 == 0:
#                 print("num: %d, right: %d" % (i, rightNum))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow([csv_name + '_lstm_2', rightNum / len(x_val)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()
#
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_lstm_3.mlmodel')
#         rightNum = 0
#         for i in range(len(x_val)):
#             result = model.predict({'input_1': x_val[i].reshape(1, 200)})
#             if np.argmax(result['Identity'][0]) == y_val[i]:
#                 rightNum += 1
#             if i % 1000 == 0:
#                 print("num: %d, right: %d" % (i, rightNum))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow([csv_name + '_lstm_3', rightNum / len(x_val)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()

    #########################################
        # model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_gru_2.mlmodel')
        # rightNum = 0
        # for i in range(len(x_val)):
        #     result = model.predict({'input_1': x_val[i].reshape(1, 200)})
        #     if np.argmax(result['Identity'][0]) == y_val[i]:
        #         rightNum += 1
        #     if i % 1000 == 0:
        #         print("num: %d, right: %d" % (i, rightNum))
        # csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
        # try:
        #     writer = csv.writer(csv_file)
        #     writer.writerow([csv_name + '_gru_2', rightNum / len(x_val)])
        #
        # finally:
        #     csv_file.close()
        # K.clear_session()
        # del model
        # gc.collect()
        #
        # model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_gru_3.mlmodel')
        # rightNum = 0
        # for i in range(len(x_val)):
        #     result = model.predict({'input_1': x_val[i].reshape(1, 200)})
        #     if np.argmax(result['Identity'][0]) == y_val[i]:
        #         rightNum += 1
        #     if i % 1000 == 0:
        #         print("num: %d, right: %d" % (i, rightNum))
        # csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
        # try:
        #     writer = csv.writer(csv_file)
        #     writer.writerow([csv_name + '_gru_3', rightNum / len(x_val)])
        #
        # finally:
        #     csv_file.close()
        # K.clear_session()
        # del model
        # gc.collect()

# for bit in bits:
#     for _ in range(1, 4):
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + 'lstm_' + str(_) + '.mlmodel')
#         rightNum = 0
#         for i in range(len(x_val)):
#             result = model.predict({'input_1': x_val[i].reshape(1, 200)})
#             if np.argmax(result['Identity'][0]) == y_val[i]:
#                 rightNum += 1
#             if i % 1000 == 0:
#                 print("num: %d, right: %d" % (i, rightNum))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow(['lstm_' + str(_), rightNum / len(x_val)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()
#
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + 'gru_' + str(_) + '.mlmodel')
#         rightNum = 0
#         for i in range(len(x_val)):
#             result = model.predict({'input_1': x_val[i].reshape(1, 200)})
#             if np.argmax(result['Identity'][0]) == y_val[i]:
#                 rightNum += 1
#             if i % 1000 == 0:
#                 print("num: %d, right: %d" % (i, rightNum))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow(['gru_' + str(_), rightNum / len(x_val)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()


#
# # QC
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

# data, label = get_QC_data("../data/train_data_pytorch.csv", "../data/test_data_pytorch.csv")
# test_indices = np.load("../QC/data/test_indices.npy")
# x_test = data[test_indices]
# y_test = label[test_indices]
# print(len(y_test))
# #
# folder_path = "../new_models/RQ3/QC/"
# save_folder = "../new_results/RQ3/QC/quantization/"
# # csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
# #              "NC", "MCP", "KMNC", "DeepGini"]
#
# csv_names = ["random"]
# # save_folder = "../new_results/RQ4/QC/quantization_"
# bits = [2, 4, 8]
# for bit in bits:
#     for csv_name in csv_names:
#         # for _ in range(20):
#         #     model = coremltools.models.MLModel(folder_path + csv_name + '_' + str(_) + '.mlmodel')
#         #     rightNum = 0
#         #     for i in range(len(x_test)):
#         #         result = model.predict({'input_1': x_test[i].reshape(1, 100)})
#         #         if np.argmax(result['Identity'][0]) == y_test[i]:
#         #             rightNum += 1
#         #         if i % 1000 == 0:
#         #             print("num: %d, right: %d" % (i, rightNum))
#         #     csv_file = open(save_folder + csv_name + '.csv', "a")
#         #     try:
#         #         writer = csv.writer(csv_file)
#         #         writer.writerow([csv_name + '_' + str(_), rightNum/len(x_test)])
#         #
#         #     finally:
#         #         csv_file.close()
#         #     K.clear_session()
#         #     del model
#         #     gc.collect()
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_gru_1.mlmodel')
#         rightNum = 0
#         for i in range(len(x_test)):
#             result = model.predict({'input_1': x_test[i].reshape(1, 100)})
#             if np.argmax(result['Identity'][0]) == y_test[i]:
#                 rightNum += 1
#             if i % 1000 == 0:
#                 print("num: %d, right: %d" % (i, rightNum))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow([csv_name + '_lstm_1', rightNum / len(x_test)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()
#
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_gru_2.mlmodel')
#         rightNum = 0
#         for i in range(len(x_test)):
#             result = model.predict({'input_1': x_test[i].reshape(1, 100)})
#             if np.argmax(result['Identity'][0]) == y_test[i]:
#                 rightNum += 1
#             if i % 1000 == 0:
#                 print("num: %d, right: %d" % (i, rightNum))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow([csv_name + '_lstm_2', rightNum / len(x_test)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()
#
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_gru_3.mlmodel')
#         rightNum = 0
#         for i in range(len(x_test)):
#             result = model.predict({'input_1': x_test[i].reshape(1, 100)})
#             if np.argmax(result['Identity'][0]) == y_test[i]:
#                 rightNum += 1
#             if i % 1000 == 0:
#                 print("num: %d, right: %d" % (i, rightNum))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow([csv_name + '_lstm_3', rightNum / len(x_test)])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()

# for bit in bits:
#     for i in range(1, 4):
#         model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + 'gru_' + str(i) + '.mlmodel')
#         rightNum = 0
#         for i in range(len(x_test)):
#             result = model.predict({'input_1': x_test[i].reshape(1, 100)})
#             if np.argmax(result['Identity'][0]) == y_test[i]:
#                 rightNum += 1
#             if i % 1000 == 0:
#                 print("num: %d, right: %d" % (i, rightNum))
#         csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
#         try:
#             writer = csv.writer(csv_file)
#             writer.writerow(['gru_' + str(i), rightNum])
#
#         finally:
#             csv_file.close()
#         K.clear_session()
#         del model
#         gc.collect()
#

# Yahoo

data, labels, texts = get_Yahoo_data()
train_index = np.load("../Yahoo/data/train_indices.npy")
test_index = np.load("../Yahoo/data/test_indices.npy")
x_train = data[train_index]
y_train = labels[train_index]
x_test = data[test_index]
y_test = labels[test_index]


folder_path = "../new_models/RQ3/Yahoo/"
save_folder = "../new_results/RQ3/Yahoo/quantization/"
csv_names = ["DSA", "LSA", "entropy", "BALD", "k_center", "margin", "entropy_dropout", "margin_dropout", "EGL",
             "NC", "MCP", "KMNC", "DeepGini"]
# csv_names = ["random"]
bits = [2, 4, 8]
for bit in bits:
    for csv_name in csv_names:
        # for _ in range(20):
        #     model = coremltools.models.MLModel(folder_path + csv_name + '_' + str(_) + '.mlmodel')
        #     rightNum = 0
        #     for i in range(len(x_test)):
        #         result = model.predict({'input_1': x_test[i].reshape(1, 1000)})
        #         if np.argmax(result['Identity'][0]) == y_test[i]:
        #             rightNum += 1
        #         if i % 500 == 0:
        #             print("num: %d, right: %d" % (i, rightNum))
        #     csv_file = open(save_folder + csv_name + '.csv', "a")
        #     try:
        #         writer = csv.writer(csv_file)
        #         writer.writerow([csv_name + '_' + str(_), rightNum/len(x_test)])
        #
        #     finally:
        #         csv_file.close()
        #     K.clear_session()
        #     del model
        #     gc.collect()

        model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_gru_1.mlmodel')
        rightNum = 0
        for i in range(len(x_test)):
            result = model.predict({'input_1': x_test[i].reshape(1, 1000)})
            if np.argmax(result['Identity'][0]) == y_test[i]:
                rightNum += 1
            if i % 500 == 0:
                print("num: %d, right: %d" % (i, rightNum))
        csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([csv_name + '_gru_1', rightNum / len(x_test)])

        finally:
            csv_file.close()
        K.clear_session()
        del model
        gc.collect()
        model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_gru_2.mlmodel')
        rightNum = 0
        for i in range(len(x_test)):
            result = model.predict({'input_1': x_test[i].reshape(1, 1000)})
            if np.argmax(result['Identity'][0]) == y_test[i]:
                rightNum += 1
            if i % 500 == 0:
                print("num: %d, right: %d" % (i, rightNum))
        csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([csv_name + '_gru_2', rightNum / len(x_test)])

        finally:
            csv_file.close()
        K.clear_session()
        del model
        gc.collect()
        model = coremltools.models.MLModel(folder_path + str(bit) + 'bit/' + csv_name + '_gru_3.mlmodel')
        rightNum = 0
        for i in range(len(x_test)):
            result = model.predict({'input_1': x_test[i].reshape(1, 1000)})
            if np.argmax(result['Identity'][0]) == y_test[i]:
                rightNum += 1
            if i % 500 == 0:
                print("num: %d, right: %d" % (i, rightNum))
        csv_file = open(save_folder + str(bit) + 'bit.csv', "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([csv_name + '_gru_3', rightNum / len(x_test)])

        finally:
            csv_file.close()
        K.clear_session()
        del model
        gc.collect()



