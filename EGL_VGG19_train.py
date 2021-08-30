from keras.datasets import cifar100
import numpy as np
from keras.models import load_model

from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from VGG19.utils import *
import keras

import csv

model = load_model("VGG19/models/EGL_VGG19.h5")

results_path = "WideResNet/results/al_EGL_wideresnet.csv"
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
    # Normalize data.
target_data = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
target_data, x_test = pre_processing(target_data, x_test)
# If subtract pixel mean is enabled
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

select_index = np.load("VGG19/data/selected_index.npy")
sore = model.evaluate(x_test, y_test)
threshold = 0.65
print("#####################################")
print(len(select_index))
print("#####################################")
if len(select_index) <= 25000 and sore[1] < threshold:

    x_train = target_data[select_index]
    y_train = y_train[select_index]

    model_checkpoint = ModelCheckpoint("VGG19/models/EGL_VGG19.h5", monitor='val_accuracy', mode='max',
                                       verbose=1, save_best_only=True,
                                       save_weights_only=False)
    # lr callback
    lr_scheduler = LearningRateScheduler(find_lr, verbose=1)
    # tensor board callback
    callbacks = [model_checkpoint, lr_scheduler]

    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(x_train)
    cifar_gen = datagen.flow(x_train, y_train, batch_size=128)

    testgen = ImageDataGenerator()
    cifar_test_gen = testgen.flow(x_test, y_test, batch_size=128)

    nb_train_samples = x_train.shape[0] // 128
    his = model.fit_generator(
        cifar_gen,
        steps_per_epoch=nb_train_samples,
        epochs=200,
        validation_data=cifar_test_gen,
        validation_steps=10000 // 128,
        callbacks=callbacks)
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


