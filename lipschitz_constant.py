from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import sys
import csv

sys.path.append("../")

# from nmutant_model.model_operation import model_load
import math
import keras


# FLAGS = flags.FLAGS

def local_lipschitz_constant(layer):
    Lipschitz_layer = 1.0
    if isinstance(layer, keras.layers.Conv2D):
        np_kernel = layer.get_weights()[0]
        shapes = np_kernel.shape
        kernel_size = shapes[0]
        input_channels = shapes[2]
        output_channels = shapes[3]
        sum_conv_max = -1
        for output_ch in range(0, output_channels):
            sum = 0
            for k1 in range(0, kernel_size):
                for k2 in range(0, kernel_size):
                    for input_ch in range(0, input_channels):
                        sum += abs(np_kernel[k1][k2][input_ch][output_ch])
            if sum > sum_conv_max:
                sum_conv_max = sum
        Lipschitz_layer = sum_conv_max
    if isinstance(layer, keras.layers.Dense):
        np_W = layer.get_weights()[0]
        input_len = np_W.shape[0]
        output_len = np_W.shape[1]
        sum_linera_max = -1
        for j in range(output_len):
            sum = 0
            for i in range(input_len):
                sum += abs(np_W[i][j])
            if sum > sum_linera_max:
                sum_linera_max = sum
        Lipschitz_layer = sum_linera_max
    if isinstance(layer, keras.layers.BatchNormalization):
        gamma = layer.gamma
        moving_variance = layer.moving_variance
        max_BN = -1
        for i in range(gamma.shape[0]):
            Lipschitz_BN = abs(gamma[i] / math.pow(moving_variance[i] + layer.epsilon, 0.5))
            if Lipschitz_BN > max_BN:
                max_BN = Lipschitz_BN

        Lipschitz_layer = max_BN
        Lipschitz_layer = Lipschitz_layer.numpy()
    return Lipschitz_layer


def global_lipschitz_constant(model, epoch=49):
    Lipschitz_model = 1.
    layers = model.layers
    for layer in layers:
        Lipschitz_model = Lipschitz_model * local_lipschitz_constant(layer)
    return Lipschitz_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-model",
                        type=str,
                        )
    parser.add_argument("--results", "-results",
                        type=str,
                        )
    args = parser.parse_args()
    model_path = args.model
    results_path = args.results
    model = keras.models.load_model(model_path)
    l = global_lipschitz_constant(model)
    csv_file = open(results_path, "a")
    try:
        writer = csv.writer(csv_file)
        writer.writerow([model_path, l])

    finally:
        csv_file.close()
    print(l)







