from keras.models import load_model
import numpy as np
import csv
import keras
from keras.utils import to_categorical

select_index = np.load("../data/IMDB_gru_selected_index.npy")
model = load_model("../../new_models/RQ1/IMDB/EGL_gru_ori.h5")
maxlen = 200
max_features = 20000
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)
target_data = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

print("#####################################")
print(len(select_index))
print("#####################################")
train_data = target_data[select_index]
train_label = y_train[select_index]
print(train_data.shape)
print(train_label.shape)
his = model.fit(train_data, train_label, batch_size=128, shuffle=True, epochs=5,
                validation_data=(x_val, y_val), verbose=1)
results_path = "../../new_results/RQ1/IMDB/EGL_gru.csv"
# test_acc = model.evaluate(x_val, y_val)[1]
test_acc = his.history['val_accuracy'][-1]
csv_file = open(results_path, "a")

try:
    writer = csv.writer(csv_file)
    writer.writerow([len(select_index), test_acc])

finally:
    csv_file.close()
model.save("../../new_models/RQ1/IMDB/EGL_gru_ori.h5")

