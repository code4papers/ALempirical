from keras.datasets import cifar10, mnist
from keras.models import load_model
import numpy as np
import keras
from art.metrics.metrics import empirical_robustness, clever_t, clever_u, clever, loss_sensitivity, wasserstein_distance
from art.estimators.classification.keras import KerasClassifier
import tensorflow as tf
import argparse
import csv
tf.compat.v1.disable_eager_execution()


def CLEVER_metric(model, data, nb_batches, batch_size, radius):
    total_res = 0
    for _ in range(len(data)):

        res0 = clever_u(model, data[_], nb_batches, batch_size, radius, norm=2, pool_factor=3)
        total_res += res0
    total_res = total_res / len(data)
    return total_res


def empirical_robustness_metric(model, data):
    params = {"eps_step": 0.1, "eps": 0.2}
    emp_robust1 = empirical_robustness(model, data, str("fgsm"), params)
    return emp_robust1


def loss_sensitivity_metric(model, data, label):
    loss1 = loss_sensitivity(model, data, label)
    return loss1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-model",
                        type=str,
                        )
    parser.add_argument("--results", "-results",
                        type=str,
                        )
    parser.add_argument("--dataset", "-dataset",
                        type=str,
                        )

    # model = load_model("NiN/NiNmodels/NiN.h5")
    args = parser.parse_args()
    model_path = args.model
    results_path = args.results
    dataset_used = args.dataset
    model = load_model(model_path)
    if dataset_used == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        y_test = keras.utils.to_categorical(y_test, 10)
        krc = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)

        # params = {"eps_step": 0.1, "eps": 0.1}
        # emp_robust1 = empirical_robustness(krc, x_test.reshape(-1, 32, 32, 3), str("fgsm"), params)
        # print(emp_robust1)
        # loss1 = loss_sensitivity(krc, x_test[-2000:].reshape(-1, 32, 32, 3), y_test[-2000:])
        # print(loss1)
        CLEVER = CLEVER_metric(krc, x_test[-500:].reshape(-1, 32, 32, 3), 10, 10, 5)
        print(CLEVER)
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, CLEVER])

        finally:
            csv_file.close()
    if dataset_used == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        krc = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
        CLEVER = CLEVER_metric(krc, x_test[-500:].reshape(-1, 28, 28, 1), 10, 10, 5)
        print(CLEVER)
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, CLEVER])

        finally:
            csv_file.close()

