import tensorly as tl
from tensorly.decomposition import partial_tucker
import numpy as np
from keras.layers import Conv2D


def tucker_decomposition(layer, k):
    """
    :param layer: weight tensor of dimensions (k,k,c,f)
    :param tucker_rank: list [r1,r2]

    :return: list of Conv2D layers [input_layer,core_layer,ouptut_layer]
            - input layer is a Conv2D layer of dimensions (1,1,c,r1)
            - core layer is a Conv2D layer of dimensions (k,k,r1,r2)
            - output layer is a Conv2D layer of dimensions (1,1,f,r1)

    """
    strides = layer.get_config()['strides']
    padding = layer.get_config()['padding']

    weights = layer.weights[0]
    print(weights.shape)
    r3 = (weights.shape[2] * k) // 8
    r4 = (weights.shape[3] * k) // 8
    bias = None
    if len(layer.get_weights()) > 1:
        bias = layer.get_weights()[1]

    # core - (k,k,r1,r2)
    # I - (c,r1)
    # O - (f,r2)
    print([r3, r4])
    core, factors = partial_tucker(weights.numpy(), modes=[2, 3], rank=[r3, r4])
    I = factors[0]
    I = np.expand_dims(I, axis=0)
    I = np.expand_dims(I, axis=0)

    O = factors[1].T
    O = np.expand_dims(O, axis=0)
    O = np.expand_dims(O, axis=0)
    # print(O.shape)
    input_layer = Conv2D(filters=r3, kernel_size=1, strides=(1, 1), padding='valid', use_bias=False)
    core_layer = Conv2D(filters=r4, kernel_size=core.shape[0], strides=strides, padding=padding, use_bias=False)
    output_layer = Conv2D(filters=O.shape[-1], kernel_size=1, strides=(1, 1), padding='valid', use_bias=True)

    input_layer.build(input_shape= [None, None, I.shape[-2]])
    core_layer.build(input_shape=[None, None, core.shape[-2]])
    output_layer.build(input_shape=[None, None, core.shape[-1]])
    # print(I.shape)
    input_layer.set_weights([I])
    core_layer.set_weights([core])
    output_layer.set_weights([O, bias])

    return [input_layer, core_layer, output_layer]


def tucker_reconstruction_loss(layer, rank):
    """
    :param: layer is a weight tensor of dimensions (k,k,c,f)
    :param: tucker_rank is a list [r1,r2]

    :return: L2 reconstruction loss for the weight matrix after tucker decomposition and reconstruction
    """
    weights = layer.weights[0]
    print(rank)
    modes = [2, 3]
    # print(weights.shape)
    # print(weights.ndim)
    core, factors = partial_tucker(weights, modes=modes, rank=rank, init='svd')

    reconstructed = core
    for i in range(len(factors)):
        reconstructed = tl.tenalg.mode_dot(reconstructed, factors[i], modes[i])

    return np.mean((weights - reconstructed) ** 2)


# def compute_rank_list(layer, k):
#     """
#     :param layer: layer is a weight tensor of dimensions (k,k,c,f)
#     :param k: a parameter varying from {0,1,...,7}
#     :return: list [r1,r2] for tucker decomposition
#             r1 = k*c/8, r2 = f*c/8
#     """
#     # c, f = layer.get_weights()[0].shape[2:]
#     r1, r2 = int(k * layer.weights[0].shape(2) // 8), int(k * layer.weights[0].shape(3) // 8)
#     # print(r1, r2)
#     return [r1, r2]
