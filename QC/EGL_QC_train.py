from keras.models import load_model
import numpy as np
import csv
import keras
from QC_model import *
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

select_index = np.load("../data/QC_lstm_selected_index.npy")
model = load_model("../../new_models/RQ1/QC/EGL_lstm_ori.h5")
data, label = get_QC_data("../../data/train_data_pytorch.csv", "../../data/test_data_pytorch.csv")
train_indices = np.load("../../QC/data/training_indices.npy")
test_indices = np.load("../../QC/data/test_indices.npy")
target_data = data[train_indices]
y_train = label[train_indices]
x_test = data[test_indices]
y_test = label[test_indices]
y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7)

print("#####################################")
print(len(select_index))
print("#####################################")
check_point = ModelCheckpoint("../../new_models/RQ1/QC/EGL_lstm_ori.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
# his = model.fit(training_data, training_label, batch_size=32, shuffle=True, epochs=epochs,
#                 validation_data=(x_val, y_val), verbose=1)
his = model.fit(target_data[select_index], y_train[select_index], batch_size=128,
                epochs=10, validation_data=(x_test, y_test),
                callbacks=[check_point])
results_path = "../../new_results/RQ1/QC/EGL_lstm.csv"
val_acc = np.max(his.history['val_accuracy'])
csv_file = open(results_path, "a")

try:
    writer = csv.writer(csv_file)
    writer.writerow([len(select_index), val_acc])

finally:
    csv_file.close()


