import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import imdb
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.utils import to_categorical


def IMDB_LSTM():
    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.25)(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def IMDB_LSTM_glove(emb_train=True):
    # embeddings_index = {}
    # with open("../data/glove.6B.100d.txt", encoding = "utf8") as f:
    #     for line in f:
    #         word, coefs = line.split(maxsplit=1)
    #         coefs = np.fromstring(coefs, "f", sep=" ")
    #         embeddings_index[word] = coefs
    #
    # print("Found %s word vectors." % len(embeddings_index))
    # word_index = imdb.get_word_index(path="imdb_word_index.json")
    # EMBEDDING_DIM = 100
    # MAX_SEQUENCE_LENGTH = 200
    # embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    # for word, i in word_index.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         # words not found in embedding index will be all-zeros.
    #         embedding_matrix[i] = embedding_vector
    #
    # embedding_layer = layers.Embedding(len(word_index) + 1,
    #                                    EMBEDDING_DIM,
    #                                    weights=[embedding_matrix],
    #                                    input_length=MAX_SEQUENCE_LENGTH,
    #                                    trainable=False)
    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    # x = embedding_layer(inputs)
    if emb_train:
        x = layers.Embedding(max_features, 128)(inputs)
    else:
        x = layers.Embedding(max_features, 128, trainable=False)(inputs)
    x = layers.SpatialDropout1D(0.2)(x)

    # Add 2 bidirectional LSTMs
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    # x = layers.Bidirectional(layers.LSTM(64))(x)
    # x = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x)
    x = layers.LSTM(128, recurrent_dropout=0.2)(x)

    x = layers.Dropout(0.2)(x)
    # Add a classifier
    x = layers.Dense(20, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def IMDB_LSTM_glove_dropout():

    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    x = layers.SpatialDropout1D(0.2)(x)

    # Add 2 bidirectional LSTMs
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    # x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.LSTM(128, recurrent_dropout=0.2)(x, training=True)
    x = layers.Dropout(0.2)(x, training=True)
    # Add a classifier
    x = layers.Dense(20, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def IMDB_LSTM_dropout():
    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.25)(x, training=True)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def IMDB_GRU():
    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(64))(x)
    x = layers.Dropout(0.25)(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def IMDB_GRU_dropout():
    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(64))(x)
    x = layers.Dropout(0.25)(x, training=True)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def IMDB_LSTM_test():
    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(200,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    # x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.LSTM(64)(x)
    # x = layers.Dropout(0.25)(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def IMDB_GRU_new(emb_train=True):

    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    # x = embedding_layer(inputs)
    if emb_train:
        x = layers.Embedding(max_features, 128)(inputs)
    else:
        x = layers.Embedding(max_features, 128, trainable=False)(inputs)
    x = layers.SpatialDropout1D(0.2)(x)

    # Add 2 bidirectional LSTMs
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    # x = layers.Bidirectional(layers.LSTM(64))(x)
    # x = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x)
    x = layers.GRU(128, recurrent_dropout=0.2)(x)

    x = layers.Dropout(0.2)(x)
    # Add a classifier
    x = layers.Dense(20, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def IMDB_GRU_new_dropout():

    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    x = layers.SpatialDropout1D(0.2)(x)

    # Add 2 bidirectional LSTMs
    # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    # x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.GRU(128, recurrent_dropout=0.2)(x, training=True)
    x = layers.Dropout(0.2)(x, training=True)
    # Add a classifier
    x = layers.Dense(20, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model

#
# model = IMDB_GRU_new(emb_train=True)
# model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
# model.summary()
# model.save("../new_models/RQ1/IMDB/EGL_gru_ori.h5")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", "-num",
                        type=int,
                        default=1
                        )
    parser.add_argument("--model_type", "-model_type",
                        type=str,
                        default='lstm'
                        )
    args = parser.parse_args()
    num = args.num
    model_type = args.model_type
    if model_type == 'lstm':
        model = IMDB_LSTM_glove(emb_train=True)
        model.summary()
        max_features = 20000
        maxlen = 200
        (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
            num_words=max_features
        )
        print(len(x_train), "Training sequences")
        print(len(x_val), "Validation sequences")
        x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
        x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
        y_train = to_categorical(y_train, 2)
        y_val = to_categorical(y_val, 2)
    #
    #     #
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        # his = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
        # print(his.history['val_accuracy'])
        model.save("../../new_models/RQ1/IMDB/lstm_" + str(num) + ".h5")
    elif model_type == 'gru':
        model = IMDB_GRU_new(emb_train=True)
        model.summary()
        max_features = 20000
        maxlen = 200
        (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
            num_words=max_features
        )
        print(len(x_train), "Training sequences")
        print(len(x_val), "Validation sequences")
        x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
        x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
        y_train = to_categorical(y_train, 2)
        y_val = to_categorical(y_val, 2)
        #
        #     #
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        his = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
        # print(his.history['val_accuracy'])
        # model.save("../../new_models/RQ1/IMDB/gru_" + str(num) + ".h5")

