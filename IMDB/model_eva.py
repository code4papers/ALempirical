import keras

model = keras.models.load_model("IMDB_models/imdb_lstm_glove.h5")
max_features = 20000
maxlen = 200
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=max_features
)

x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
y_val = keras.utils.to_categorical(y_val, 2)
print(model.evaluate(x_val, y_val)[1])
