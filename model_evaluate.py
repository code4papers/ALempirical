from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np

# model = load_model("models/BALD_lenet_5.h5")
# model.summary()

(_, _), (_, y_test) = mnist.load_data()
# x_test = x_test.reshape(-1, 28, 28, 1)
# x_test = x_test.astype('float32') / 255

y_test = to_categorical(y_test, 10)

x_test_1 = np.load("data/mnist_brightness.npy").astype('float32') / 255
x_test_2 = np.load("data/mnist_contrast.npy").astype('float32') / 255
x_test_3 = np.load("data/mnist_rotation.npy").astype('float32') / 255
x_test_4 = np.load("data/mnist_shear.npy").astype('float32') / 255
x_test_1 = x_test_1.reshape(-1, 28, 28, 1)
x_test_2 = x_test_2.reshape(-1, 28, 28, 1)
x_test_3 = x_test_3.reshape(-1, 28, 28, 1)
x_test_4 = x_test_4.reshape(-1, 28, 28, 1)

model = load_model("models/lenet-5.h5")
brightness_score = model.evaluate(x_test_1, y_test)
contrast_score = model.evaluate(x_test_2, y_test)
rotation_score = model.evaluate(x_test_3, y_test)
shear_score = model.evaluate(x_test_4, y_test)

print("ori model, brightness acc:", brightness_score[1])
print("ori model, contrast acc:", contrast_score[1])
print("ori model, rotation acc:", rotation_score[1])
print("ori model, shear acc:", shear_score[1])


model = load_model("models/al_lenet_5_2.h5")
brightness_score = model.evaluate(x_test_1, y_test)
contrast_score = model.evaluate(x_test_2, y_test)
rotation_score = model.evaluate(x_test_3, y_test)
shear_score = model.evaluate(x_test_4, y_test)

print("entropy model, brightness acc:", brightness_score[1])
print("entropy model, contrast acc:", contrast_score[1])
print("entropy model, rotation acc:", rotation_score[1])
print("entropy model, shear acc:", shear_score[1])

model = load_model("models/BALD_lenet_5.h5")
brightness_score = model.evaluate(x_test_1, y_test)
contrast_score = model.evaluate(x_test_2, y_test)
rotation_score = model.evaluate(x_test_3, y_test)
shear_score = model.evaluate(x_test_4, y_test)

print("BALD model, brightness acc:", brightness_score[1])
print("BALD model, contrast acc:", contrast_score[1])
print("BALD model, rotation acc:", rotation_score[1])
print("BALD model, shear acc:", shear_score[1])
