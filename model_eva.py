from keras.datasets import cifar10
from keras.models import load_model
import numpy as np
import keras


model = load_model("models/res20.h5")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train_mean = np.mean(x_train, axis=0)
# x_train -= x_train_mean
x_test -= x_train_mean
y_test = keras.utils.to_categorical(y_test, 10)
score = model.evaluate(x_test, y_test)
print("test acc: ", score[1])
