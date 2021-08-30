from keras.models import load_model
import numpy as np
import csv
import keras
from Yahoo_model import *
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

threshold = 0.86
select_index = np.load("../data/Yahoo_lstm_selected_index.npy")
model = load_model("../../new_models/RQ1/Yahoo/EGL_lstm_ori.h5")
data, labels, texts = get_Yahoo_data()
train_index = np.load("../../Yahoo/data/train_indices.npy")
test_index = np.load("../../Yahoo/data/test_indices.npy")
target_data = data[train_index]
y_train = labels[train_index]
x_test = data[test_index]
y_test = labels[test_index]

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
score = model.evaluate(x_test, y_test)

print("#####################################")
print(len(select_index))
print("#####################################")
check_point = ModelCheckpoint("../../new_models/RQ1/Yahoo/EGL_lstm_ori.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
# his = model.fit(training_data, training_label, batch_size=32, shuffle=True, epochs=epochs,
#                 validation_data=(x_val, y_val), verbose=1)
his = model.fit(target_data[select_index], y_train[select_index], batch_size=32,
                epochs=16, validation_data=(x_test, y_test),
                callbacks=[check_point])
results_path = "../../new_results/RQ1/Yahoo/EGL_lstm.csv"
val_acc = np.max(his.history['val_accuracy'])
csv_file = open(results_path, "a")

try:
    writer = csv.writer(csv_file)
    writer.writerow([len(select_index), val_acc])

finally:
    csv_file.close()


