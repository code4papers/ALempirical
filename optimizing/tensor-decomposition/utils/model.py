# Model related imports
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from .tucker import tucker_decomposition


def decomposed_model(model, k):
    decomposed_model = Sequential()

    for (index, layer) in list(enumerate(model.layers)):
        if isinstance(layer, Conv2D) and (index > 3):
            #if index == 0:
            #    decomposed_model.add(layer)
            #    decomposed_model.layers[-1].from_config(model.layers[index].get_config())
            #    decomposed_model.layers[-1].set_weights(model.layers[index].get_weights())

            #else:
            input_layer, core_layer, output_layer = tucker_decomposition(layer,
                                                                         k)

            decomposed_model.add(input_layer)
            decomposed_model.add(core_layer)
            decomposed_model.add(output_layer)

        else:
            decomposed_model.add(layer)
            decomposed_model.layers[-1].from_config(model.layers[index].get_config())
            decomposed_model.layers[-1].set_weights(model.layers[index].get_weights())

    return decomposed_model
