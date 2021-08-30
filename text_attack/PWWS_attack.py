import keras
from keras.utils import to_categorical
from adversarial_tools import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import csv
from pwws_utils import *
import argparse


def get_tokenizer():
    max_features = 10000
    max_len = 100
    texts = []
    labels = []
    train_data_path = '../../data/train_data_pytorch.csv'
    test_data_path = '../../data/test_data_pytorch.csv'
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
    return tokenizer, texts, data, labels


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
    return tokenizer, data, labels, texts


def fool_text_classifier(model_path, data_type, save_path, data_select):
    # clean_samples_cap = 10
    dataset = data_type
    # clean_samples_cap = clean_samples_cap  # 1000
    # print('clean_samples_cap:', clean_samples_cap)

    # get tokenizer
    dataset = dataset

    # Read data set
    x_test = y_test = None
    test_texts = None
    if dataset == 'imdb':
        max_features = 20000
        maxlen = 200
        (x_train, y_train), (x_val, y_test) = keras.datasets.imdb.load_data(
            num_words=max_features
        )
        print(len(x_train), "Training sequences")
        print(len(x_val), "Validation sequences")
        x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
        test_texts = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
        y_train = to_categorical(y_train, 2)
        y_test = to_categorical(y_test, 2)
        INDEX_FROM = 3
        tokenizer = keras.datasets.imdb.get_word_index()
        tokenizer = {k: (v + INDEX_FROM) for k, v in tokenizer.items()}

        tokenizer["<PAD>"] = 0
        tokenizer["<START>"] = 1
        tokenizer["<UNK>"] = 2
        tokenizer = {k: v for k, v in sorted(tokenizer.items(), key=lambda item: item[1])}
        tokenizer = {k: tokenizer[k] for k in list(tokenizer)[:20000]}
        # print(tokenizer)
    elif dataset == 'qc':
        tokenizer, texts, index_data, labels = get_tokenizer()
        test_indices = np.load("../../QC/data/test_indices.npy")
        texts = np.asarray(texts)
        # print(texts)
        # print(test_indices)
        test_texts = texts[test_indices]
        x_test = index_data[test_indices]
        y_test = labels[test_indices]
        y_test = to_categorical(y_test, 7)
    elif dataset == 'yahoo':
        tokenizer, data, labels, texts = get_Yahoo_data()
        train_index = np.load("../../Yahoo/data/train_indices.npy")
        test_index = np.load("../../Yahoo/data/test_indices.npy")

        x_train = data[train_index]
        y_train = labels[train_index]
        x_test = data[test_index]
        y_test = labels[test_index]
        texts = np.asarray(texts)
        test_texts = texts[test_index]
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)


    # Write clean examples into a txt file
    # clean_texts_path = r'./fool_result/{}/clean_{}.txt'.format(dataset, str(clean_samples_cap))
    # if not os.path.isfile(clean_texts_path):
    #     write_origin_input_texts(clean_texts_path, test_texts)

    # Select the model and load the trained weights
    model = keras.models.load_model(model_path)
    model.summary()
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    test_texts = np.asarray(test_texts)
    x_test = x_test[data_select]
    y_test = y_test[data_select]
    test_texts = test_texts[data_select]
    # evaluate classification accuracy of model on clean samples
    scores_origin = model.evaluate(x_test, y_test)
    print('clean samples origin test_loss: %f, accuracy: %f' % (scores_origin[0], scores_origin[1]))
    # all_scores_origin = model.evaluate(x_test, y_test)
    # print('all origin test_loss: %f, accuracy: %f' % (all_scores_origin[0], all_scores_origin[1]))

    grad_guide = ForwardGradWrapper(model)
    classes_prediction = grad_guide.predict_classes(x_test)

    print('Crafting adversarial examples...')
    successful_perturbations = 0
    failed_perturbations = 0
    sub_rate_list = []
    NE_rate_list = []

    adv_text_path = "test_adv.txt"
    change_tuple_path = "test_change.txt"
    # file_1 = open(adv_text_path, "a")
    # file_2 = open(change_tuple_path, "a")
    for index, text in enumerate(test_texts):
        sub_rate = 0
        NE_rate = 0
        # text = text.tolist()
        if dataset == 'imdb':
            text = index2text(text, tokenizer)
        if dataset == 'qc':
            text = text.tolist()
        if dataset == 'yahoo':
            text = text.tolist()
        if np.argmax(y_test[index]) == classes_prediction[index]:
            # If the ground_true label is the same as the predicted label
            adv_doc, adv_y, sub_rate, NE_rate, change_tuple_list = adversarial_paraphrase(input_text=text,
                                                                                          true_y=np.argmax(y_test[index]),
                                                                                          grad_guide=grad_guide,
                                                                                          tokenizer=tokenizer,
                                                                                          dataset=dataset,
                                                                                          level='word')
            if adv_y != np.argmax(y_test[index]):
                successful_perturbations += 1
                print('{}. Successful example crafted.'.format(index))
            else:
                failed_perturbations += 1
                print('{}. Failure.'.format(index))

            text = adv_doc
            sub_rate_list.append(sub_rate)
            NE_rate_list.append(NE_rate)
        #     file_2.write(str(index) + str(change_tuple_list) + '\n')
        # file_1.write(text + " sub_rate: " + str(sub_rate) + "; NE_rate: " + str(NE_rate) + "\n")

    success_rate = successful_perturbations / (successful_perturbations + failed_perturbations)
    mean_sub_rate = sum(sub_rate_list) / len(sub_rate_list)
    mean_NE_rate = sum(NE_rate_list) / len(NE_rate_list)
    print('mean substitution rate:', mean_sub_rate)
    print('mean NE rate:', mean_NE_rate)
    print('Success rate:', success_rate)
    csv_file = open(save_path, "a")
    try:
        writer = csv.writer(csv_file)
        writer.writerow([model_path, mean_sub_rate, mean_NE_rate, success_rate])

    finally:
        csv_file.close()
    # file_1.close()
    # file_2.close()

#
# def index2text(index_input):
#     INDEX_FROM = 3
#     word_to_id = keras.datasets.imdb.get_word_index()
#     # print(word_to_id)
#     word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
#     word_to_id["<PAD>"] = 0
#     word_to_id["<START>"] = 1
#     word_to_id["<UNK>"] = 2
#     id_to_word = {value: key for key, value in word_to_id.items()}
#     print(' '.join(id_to_word[id] for id in index_input))

# index2text(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        )
    parser.add_argument("--data_type", "-data_type",
                        type=str,
                        )
    parser.add_argument("--result_path", "-result_path",
                        type=str,
                        )
    parser.add_argument("--select_path", "-select_path",
                        type=str,
                        )
    args = parser.parse_args()
    model_path = args.model_path
    result_path = args.result_path
    data_type = args.data_type
    select_path = args.select_path
    if data_type == 'imdb':
        select_data = np.load(select_path)
    elif data_type == 'qc':
        select_data = np.arange(1000)
        # select_data = select_data.tolist()
        # print(select_data)
    elif data_type == 'yahoo':
        select_data = np.arange(200)

    fool_text_classifier(model_path, data_type, result_path, select_data)

# model_path = "../IMDB/IMDB_models/imdb_lstm_glove.h5"
# fool_text_classifier(model_path, 'imdb', "test.csv")
