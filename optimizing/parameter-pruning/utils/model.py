# Model related imports
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from .pruned_layers import pruned_Conv2D, pruned_Dense


def get_model(args):
    BN_alpha = args.BN_alpha
    BN_eps = args.BN_eps

    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', input_shape=[32, 32, 3]))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=BN_alpha, epsilon=BN_eps))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


# def convert_to_masked_model(model):
#     """x
#     :param model: input model structure as Sequential structure
#     :return: another model with masked_conv and masked_dense layers
#     """
#     masked_model = Sequential()
#
#     for (index, layer) in list(enumerate(model.layers)):
#         if index == 0:
#             if isinstance(layer, Conv2D):
#                 conf = layer.__class__.get_config(layer)
#                 if index == 0:
#                     pruned_layer = pruned_Conv2D(filters=conf['filters'], kernel_size=conf['kernel_size'], strides=conf['strides'],
#                                       padding=conf['padding'], input_shape=list(conf['batch_input_shape'][1:]), name="prune" + str(index))
#                     # pruned_layer.set_config(layer.get_config())
#                     # pruned_layer.build(input_shape=curr_shape)
#                     pruned_layer.set_weights(layer.get_weights() + [pruned_layer.get_mask()])
#                     masked_model.add(pruned_layer)
#                     # masked_model.add(
#                     #     pruned_Conv2D(filters=conf['filters'], kernel_size=conf['kernel_size'], strides=conf['strides'],
#                     #                   padding=conf['padding'], input_shape=list(conf['batch_input_shape'][1:]), name="prune" + str(index)))
#                 else:
#                     pruned_layer = pruned_Conv2D(filters=conf['filters'], kernel_size=conf['kernel_size'], strides=conf['strides'],
#                                       padding=conf['padding'], name="prune" + str(index))
#                     pruned_layer.set_weights(layer.get_weights() + [pruned_layer.get_mask()])
#                     masked_model.add(pruned_layer)
#                     # masked_model.add(
#                     #     pruned_Conv2D(filters=conf['filters'], kernel_size=conf['kernel_size'], strides=conf['strides'],
#                     #                   padding=conf['padding'], name="prune" + str(index)))
#
#             elif isinstance(layer, Dense):
#                 conf = layer.__class__.get_config(layer)
#                 pruned_layer = pruned_Dense(conf['units'])
#                 pruned_layer.set_weights(layer.get_weights() + [pruned_layer.get_mask()])
#
#                 # masked_model.add(pruned_Dense(conf['units']))
#
#             else:
#                 masked_model.add(layer)
#     # print("##########################")
#     # print(len(masked_model.layers))
#     # print(len(model.layers))
#     # print("##########################")
#     # a = 1
#     # for masked_layer, model_layer in zip(masked_model.layers, model.layers):
#     #     if a == 1:
#     #         if isinstance(model_layer, Conv2D) or isinstance(model_layer, Dense):
#     #             a += 1
#     #             new_weights = model_layer.get_weights()
#     #             # new_weights.append(masked_layer.get_weights()[-1])
#     #             # print(masked_layer.get_weights()[-1])
#     #             masked_layer.set_weights(model_layer.get_weights() + [masked_layer.get_mask()])
#     #         else:
#     #             masked_layer.from_config(model_layer.get_config())
#     #             masked_layer.set_weights(model_layer.get_weights())
#
#     return masked_model


def convert_to_masked_model(model):
    prev_layer = None
    # implement a function that takes a model and returns another model with masked_conv and masked_dense layers
    ret_model = Sequential()
    for i, layer in enumerate(model.layers):

        if i == 0:
            curr_shape = model.input_shape
            prev_layer = layer
        else:
            curr_shape = prev_layer.compute_output_shape(curr_shape)

        if isinstance(layer, Conv2D):
            pruned_layer = pruned_Conv2D(filters=layer.get_config()['filters'],
                                         kernel_size=layer.get_config()['kernel_size'])  # , strides=(1, 1), padding='valid',input_shape=[32,32,3]))
            pruned_layer.set_config(layer.get_config())
            pruned_layer.build(input_shape=curr_shape)
            # print(pruned_layer.get_config())
            # print(layer.get_config())
            print(pruned_layer.get_mask().shape)
            print(layer.get_weights()[0].shape)
            pruned_layer.set_weights(layer.get_weights() + [pruned_layer.get_mask()])

        elif isinstance(layer, Dense):
            pruned_layer = pruned_Dense(n_neurons_out=layer.get_config()['units'])
            pruned_layer.set_config(layer.get_config())
            pruned_layer.build(input_shape=curr_shape)
            pruned_layer.set_weights(layer.get_weights() + [pruned_layer.get_mask()])

        else:
            pruned_layer = layer

        ret_model.add(pruned_layer)

        prev_layer = layer
    # pruned_layer.set_weights(layer.get_weights())
    return ret_model
