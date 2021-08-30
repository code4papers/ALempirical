from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten, Dropout
from keras.models import Model
import keras
from keras.datasets import mnist

def Lenet5():
    # ori acc 0.9889
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)
    img_rows, img_cols = 28, 28
    input_tensor = Input(shape=(img_rows, img_cols, 1))

    # block1
    x = Convolution2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)
    return model


def Lenet5_dropout():
    # ori acc 0.9889
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)
    img_rows, img_cols = 28, 28
    input_tensor = Input(shape=(img_rows, img_cols, 1))

    # block1
    x = Convolution2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)
    return model


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# y_test = keras.utils.to_categorical(y_test, 10)
# y_train = keras.utils.to_categorical(y_train, 10)
#
# model = Lenet5()
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.summary()
# his = model.fit(x_train, y_train, batch_size=256, shuffle=True, epochs=20,
#                 validation_data=(x_test, y_test), verbose=1)
#
# print(his.history['val_accuracy'])
# model.save("new_models/RQ1/Lenet5/Lenet5_2.h5")

if __name__ == '__main__':
    model = Lenet5_dropout()
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    print(model.get_config())
    print(model.layers[-5].get_config())
    print(model.predict(x_test[0].reshape(1, 28, 28, 1)))
    print(model.predict(x_test[0].reshape(1, 28, 28, 1)))
    print(model.predict(x_test[0].reshape(1, 28, 28, 1)))
