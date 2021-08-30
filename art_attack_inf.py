import art
from keras.datasets import mnist, cifar10, cifar100
import numpy as np
import csv
from keras.models import load_model
import argparse
import tensorflow as tf
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, SaliencyMapMethod, CarliniL2Method
from check_art import *

tf.compat.v1.disable_eager_execution()

def color_preprocessing(x_train, x_test):

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def art_attack(eps=8/255, eps_step=8/2550):
    model_path = "VGG/VGGmodels/VGG16_new.h5"
    data_type = 'cifar10'
    attack_type = 'pgd'
    save_path = "VGG/results/pgd_inf.csv"
    if data_type == 'cifar10':
        model = load_model(model_path)
        classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
        if attack_type == 'pgd':
            attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=eps, eps_step=eps_step, max_iter=100)
        (x_train, _), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        # x_train_mean = np.mean(x_train, axis=0)
        # x_test -= x_train_mean
        y_test = y_test.reshape(1, -1)[0]
        split = int(len(x_test) / 100)
        adv_num = 0
        total_num = 0
        for _ in range(100):
            x_part = x_test[split * _: split * (_ + 1)]
            y_part = y_test[split * _: split * (_ + 1)]
            ori_predictions = model.predict(x_part.reshape(-1, 32, 32, 3))
            ori_label = np.argmax(ori_predictions, axis=1)
            correct_classified = np.where(ori_label == y_part)[0]
            x_part = x_part[correct_classified]
            y_part = y_part[correct_classified]
            x_test_adv = attack.generate(x=x_part)
            predictions = model.predict(x_test_adv)
            adv_num += np.sum(np.argmax(predictions, axis=1) != y_part)
            total_num += len(y_part)
            # print(adv_num)
            # print(total_num)
        robust_accuracy = adv_num / total_num
        print(robust_accuracy)

        csv_file = open(save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, attack_type, robust_accuracy])

        finally:
            csv_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", "-eps",
                        type=float,
                        )
    parser.add_argument("--eps_step", "-eps_step",
                        type=float,
                        )

    args = parser.parse_args()
    eps = args.eps
    eps_step = args.eps_step
    art_attack(eps=eps, eps_step=eps_step)










