from keras.datasets import cifar10
import numpy as np
from keras.models import load_model

from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import keras

import csv
model = load_model("../../new_models/RQ1/VGG16/EGL.h5")

results_path = "../../new_results/RQ1/VGG16/EGL.csv"
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalize data.
target_data = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

select_index = np.load("../data/VGG16_selected_index.npy")
print("#####################################")
print(len(select_index))
print("#####################################")

x_train = target_data[select_index]
y_train = y_train[select_index]

lr = 1e-2

def lr_scheduler(epoch):
    initial_lrate = lr
    drop = 0.9
    epochs_drop = 50.0
    lrate = initial_lrate * np.power(drop,
                                     np.floor((1 + epoch) / epochs_drop))
    return lrate

reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
# datagen.fit(x_train)
lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=128)
bestmodelname = "../../new_models/RQ1/VGG16/EGL.h5"
checkPoint = keras.callbacks.ModelCheckpoint(bestmodelname, monitor="val_accuracy", save_best_only=True, verbose=1)
nb_train_samples = x_train.shape[0] // 128
nb_epoch = 200
his = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=nb_epoch,
    validation_data=(x_test, y_test),
    callbacks=[checkPoint, reduce_lr])
print("training.....")
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


