from keras.datasets import cifar10
import numpy as np
from keras.models import load_model

from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import keras

import csv
model = load_model("../../new_models/RQ1/NiN/EGL_for_at.h5")

results_path = "../../new_results/RQ1/NiN/EGL_for_at.csv"
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalize data.
target_data = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
x_train_mean = np.mean(target_data, axis=0)
target_data -= x_train_mean
x_test -= x_train_mean
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

select_index = np.load("../data/NiN_selected_index.npy")
threshold = 86.43
print("#####################################")
print(len(select_index))
print("#####################################")

x_train = target_data[select_index]
y_train = y_train[select_index]
# datagen.fit(x_train)
lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
callbacks = [lr_reducer, early_stop]
model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
print("training.....")
print("training length: ", len(x_train))

his = model.fit(x_train, y_train,
                  batch_size=128,
                  epochs=200,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)

print("training over...")

val_acc = his.history['val_accuracy'][-1]
csv_file = open(results_path, "a")
try:
    writer = csv.writer(csv_file)
    writer.writerow([len(select_index), val_acc])

finally:
    csv_file.close()


model.save("../../new_models/RQ1/NiN/EGL_for_at.h5")

