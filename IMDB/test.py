from keras.models import load_model
import keras
from keras.utils import to_categorical

model = load_model("IMDB_models/EGL_imdb_lstm_new.h5")
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
score = model.evaluate(x_val, y_val)
print(score[1])
