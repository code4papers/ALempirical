from keras.datasets import cifar100
import numpy as np
from keras.models import load_model

from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras

import csv

def color_preprocessing(x_train, x_test):

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test

model = load_model("WideResNet/models/EGL_wideresnet.h5")

results_path = "WideResNet/results/al_EGL_wideresnet.csv"
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
    # Normalize data.
target_data = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
target_data, x_test = color_preprocessing(target_data, x_test)
# If subtract pixel mean is enabled
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

select_index = np.load("WideResNet/data/selected_index.npy")
sore = model.evaluate(x_test, y_test)
threshold = 0.7
print("#####################################")
print(len(select_index))
print("#####################################")
if len(select_index) <= 25000 and sore[1] < threshold:

    x_train = target_data[select_index]
    y_train = y_train[select_index]

    lr = 0.1
    def lr_scheduler(epoch):
        initial_lrate = lr
        if 60 <= epoch < 120:
            initial_lrate = 0.02
        if 120 <= epoch < 160:
            initial_lrate = 0.004
        if epoch > 160:
            initial_lrate = 0.0008
        return initial_lrate


    generator = ImageDataGenerator(
        vertical_flip=True,
        horizontal_flip=True
    )

    generator.fit(x_train)
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    bestmodelname = "WideResNet/models/EGL_wideresnet.h5"
    early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=20)
    model_checkpoint = ModelCheckpoint(bestmodelname, monitor="val_accuracy",
                                       save_best_only=True)

    train_generator = generator.flow(x_train, y_train, batch_size=128)
    nb_train_samples = x_train.shape[0] // 128
    his = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples,
        epochs=200,
        validation_data=(x_test, y_test),
        validation_steps=10000 // 128,
        callbacks=[model_checkpoint, reduce_lr, early_stopper])
    print("training length: ", len(x_train))
    # generator.fit(x_train)
    # train_generator = generator.flow(x_train, y_train, batch_size=100)
    # his = model.fit(train_generator,
    #                 validation_data=(x_test, y_test),
    #                 epochs=200, verbose=0, workers=4,
    #                 callbacks=callbacks)

    print("training over...")


    val_acc = np.max(his.history['val_accuracy'])
    csv_file = open(results_path, "a")
    try:
        writer = csv.writer(csv_file)
        writer.writerow([len(select_index), val_acc])

    finally:
        csv_file.close()


