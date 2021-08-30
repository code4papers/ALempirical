import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd

import argparse
# import model
import scoring
# import scoring_char
import transformer
# import transformer_char
import numpy as np
import pickle
import keras
import csv
import sys

np.random.seed(7)

# print(model)
# model = keras.models.load_model("../QC/QCmodels/QC_lstm.h5")
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


def recoveradv(rawsequence, index2word, inputs, advwords):
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n '
    rear_ct = len(rawsequence)
    advsequence = rawsequence[:]
    try:
        for i in range(inputs.size()[0] - 1, -1, -1):
            wordi = index2word[inputs[i].item()]
            rear_ct = rawsequence[:rear_ct].rfind(wordi)
            # print(rear_ct)
            if inputs[i].item() >= 3:
                advsequence = advsequence[:rear_ct] + advwords[i] + advsequence[rear_ct + len(wordi):]
    except:
        print('something went wrong')
    return advsequence


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


def attackword(dataset, model_path, result_path, power=5, data_select=None, maxbatch=None):
    model = keras.models.load_model(model_path)
    if dataset == 'qc':
        numclass = 7
        maxlen = 100
        tokenizer, texts, index_data, labels = get_tokenizer()
        word_index = tokenizer.word_index
        index2word = {value: key for key, value in word_index.items()}
        test_indices = np.load("../../QC/data/test_indices.npy")
        texts = np.asarray(texts)
        test_texts = texts[test_indices]
        x_test = index_data[test_indices]
        y_test = labels[test_indices]
        y_test = to_categorical(y_test, numclass)
        x_test = x_test[data_select]
        y_test = y_test[data_select]
        test_texts = test_texts[data_select]
        word_index = tokenizer.word_index
        dictionarysize = 10000

    elif dataset == 'imdb':

        INDEX_FROM = 3
        tokenizer = keras.datasets.imdb.get_word_index()
        tokenizer = {k: (v + INDEX_FROM) for k, v in tokenizer.items()}

        tokenizer["<PAD>"] = 0
        tokenizer["<START>"] = 1
        tokenizer["<UNK>"] = 2
        tokenizer = {k: v for k, v in sorted(tokenizer.items(), key=lambda item: item[1])}
        tokenizer = {k: tokenizer[k] for k in list(tokenizer)[:20000]}
        index2word = {value: key for key, value in tokenizer.items()}

        max_features = 20000
        maxlen = 200
        numclass = 2
        (x_train, y_train), (x_val, y_test) = keras.datasets.imdb.load_data(
            num_words=max_features
        )
        print(len(x_train), "Training sequences")
        print(len(x_val), "Validation sequences")
        # x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
        y_test = to_categorical(y_test, numclass)
        x_test = x_test[data_select]
        y_test = y_test[data_select]
        word_index = keras.datasets.imdb.get_word_index()
        dictionarysize = 20000

    elif dataset == 'yahoo':
        numclass = 10
        maxlen = 1000
        tokenizer, data, labels, texts = get_Yahoo_data()
        word_index = tokenizer.word_index
        index2word = {value: key for key, value in word_index.items()}
        train_index = np.load("../../Yahoo/data/train_indices.npy")
        test_index = np.load("../../Yahoo/data/test_indices.npy")
        x_train = data[train_index]
        y_train = labels[train_index]
        x_test = data[test_index]
        y_test = labels[test_index]

        y_train = to_categorical(y_train, numclass)
        y_test = to_categorical(y_test, numclass)
        x_test = x_test[data_select]
        y_test = y_test[data_select]
        # test_texts = test_texts[data_select]
        dictionarysize = 20000

    corrects = .0
    tgt = []
    adv = []
    origsample = []
    origsampleidx = []
    # numclass = 7
    # word_index = tokenizer.word_index

    for dataid in range(len(x_test)):
        print(dataid)
        if maxbatch != None and dataid >= maxbatch:
            break
        # inputs, target, idx, raw = data
        inputs = x_test[dataid].reshape(-1, maxlen)
        # print(inputs.shape)
        target = y_test[dataid]
        idx = dataid
        # raw = test_texts[dataid]
        # if dataset == 'qc' and 'yahoo':
        #     raw = test_texts[dataid]
        # else:
        #     raw = index2text(x_test[dataid], tokenizer)
        # print(raw)
        # inputs, target = inputs.to(device), target.to(device)
        origsample.append(inputs)
        origsampleidx.append(idx)
        tgt.append(target)
        wtmp = []

        output = model.predict(inputs)
        losses = scoring.scorefunc("replaceone")(model, inputs, output, numclass)[0]
        # print(losses)

        indices = np.argsort(-losses)
        advinputs = inputs.copy()

        for k in range(inputs.shape[0]):
            wtmp.append([])
            for i in range(inputs.shape[1]):
                # print(advinputs[k, i].item())
                if advinputs[k, i].item() > 3:
                    wtmp[-1].append(index2word[advinputs[k, i].item()])
                else:
                    wtmp[-1].append('')
        # print(wtmp)
        for k in range(inputs.shape[0]):
            j = 0
            t = 0
            while j < power and t < inputs.shape[1]:
                if advinputs[k, indices[t]].item() > 3:
                    word, advinputs[k, indices[t]] = transformer.transform('homoglyph')(
                        advinputs[k, indices[t]].item(), word_index, index2word, top_words=dictionarysize)
                    wtmp[k][indices[t]] = word
                    j += 1
                t += 1
        adv.append(advinputs)

        output2 = model.predict(advinputs)
        pred2 = np.argmax(output2, axis=1)[0]
        # print(pred2)
        # print(np.argmax(target))
        if pred2 == np.argmax(target):
            corrects += 1

    acc = corrects / len(adv)
    # print(adv.shape[0])
    print('attack success rate%.5f' % (1 - acc))
    csv_file = open(result_path, "a")
    try:
        writer = csv.writer(csv_file)
        writer.writerow([model_path, 1 - acc])

    finally:
        csv_file.close()


def index2text(index_input, tokenizer):
    id_to_word = {value: key for key, value in tokenizer.items()}

    index_input = [x for x in index_input if x != 0 and x != 1 and x != 2]
    # print(' '.join(id_to_word[id] for id in index_input))
    return ' '.join(id_to_word[id] for id in index_input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data')
    parser.add_argument('--dataset', '-dataset', type=str, default='qc')
    parser.add_argument('--model_path', '-model_path', type=str)
    parser.add_argument('--save_path', '-save_path', type=str)
    parser.add_argument('--data_select', '-data_select', type=str, default=None)

    args = parser.parse_args()
    dataset = args.dataset
    model_path = args.model_path
    save_path = args.save_path
    data_select = args.data_select
    if dataset == 'qc':
        select_data = np.arange(1000)
        attackword(dataset, model_path, save_path, data_select=select_data)
    elif dataset == 'yahoo':
        select_data = np.arange(200)
        attackword(dataset, model_path, save_path, data_select=select_data)
    else:
        data_select = np.load(data_select)
        attackword(dataset, model_path, save_path, data_select=data_select)


# attackword('imdb', "../IMDB/IMDB_models/imdb_lstm_glove.h5", data_path="../IMDB/data/CLEVER_data1.npy")



