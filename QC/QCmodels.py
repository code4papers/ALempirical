from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dropout, Dense, Input
import os
import keras
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np


def get_QC_data(data_path):
    max_features = 10000
    max_len = 100
    texts = []
    labels = []
    qc = pd.read_csv(data_path,
                     names=['num', 'title', 'description', 'category'])
    for i in range(len(qc)):
        texts.append(str(qc['description'][i]))
        labels.append(qc['category'][i])

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_len)
    labels = np.asarray(labels)
    return data, labels




max_features = 10000
time_steps = 100
validate_indices = 0.2

data_path = '../data/train_data_pytorch.csv'
BASE_DIR = '../data/'
# GLOVE_DIR = os.path.join(BASE_DIR, '.vector_cache')
# print('Indexing word vectors.')
embeddings_index = {}
with open("../data/glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        # line = line.decode('utf-8')
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))
texts = []
labels = []
qc = pd.read_csv(data_path,
                 names=['num', 'title', 'description', 'category'])
for i in range(len(qc)):
    texts.append(str(qc['description'][i]))
    labels.append(qc['category'][i])

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
EMBEDDING_DIM = 100 # how big is each word vector

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=time_steps,
                            trainable=True)

inputs = Input(shape=(100,), dtype="int32")
x = embedding_layer(inputs)
x = keras.layers.Conv1D(128, 5, activation='relu')(x)
x = keras.layers.MaxPool1D(2)(x)
x = LSTM(60)(x)
# x = Dropout(0.5)(x)
# x = LSTM(60, recurrent_dropout=0.5, name="ls2")(x)
# x = Dense(60, activation='relu')(x)
outputs = Dense(7, activation='softmax')(x)
model = keras.Model(inputs, outputs)
model.summary()


x_train, y_train = get_QC_data('../data/train_data_pytorch.csv')
x_test, y_test = get_QC_data("../data/test_data_pytorch.csv")
y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
print(x_train.shape)
x_validation = x_train[-2808:]
y_validation = y_train[-2808:]
x_train = x_train[:20000]
y_train = y_train[:20000]

model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
his = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_validation, y_validation))
score = model.evaluate(x_test, y_test)
print(his.history['val_accuracy'])
print("test acc: ", score[1])


