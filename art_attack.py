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


def art_attack(model_path, model_type, attack_type, para, save_path):

    if model_type == 'lenet':
        model = load_model(model_path)
        classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
        if attack_type == 'fgsm':
            attack = FastGradientMethod(estimator=classifier, norm=np.inf, eps=para)
        elif attack_type == 'pgd':
            attack = ProjectedGradientDescent(estimator=classifier, eps=1., eps_step=0.1, max_iter=100)
        elif attack_type == 'jsma':
            attack = SaliencyMapMethod(classifier=classifier, theta=1., gamma=para)
        elif attack_type == 'cw':
            attack = CarliniL2Method(classifier=classifier, initial_const=para, max_iter=50, binary_search_steps=1, batch_size=5, learning_rate=0.1)
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype('float32') / 255
        x_test = x_test.reshape(-1, 28, 28, 1)
        split = int(len(x_test) / 100)
        adv_num = 0
        total_num = 0
        for _ in range(100):
            x_part = x_test[split * _: split * (_ + 1)]
            y_part = y_test[split * _: split * (_ + 1)]
            ori_predictions = model.predict(x_part.reshape(-1, 28, 28, 1))
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
    elif model_type == 'NiN' or model_type == 'VGG16':
        model = load_model(model_path)
        classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
        if attack_type == 'fgsm':
            attack = FastGradientMethod(estimator=classifier, norm=2, eps=para)
        elif attack_type == 'pgd':
            attack = ProjectedGradientDescent(estimator=classifier, eps=0.2, eps_step=0.1, max_iter=100)
        elif attack_type == 'jsma':
            attack = SaliencyMapMethod(classifier=classifier, theta=1., gamma=para)
        elif attack_type == 'cw':
            attack = CarliniL2Method(classifier=classifier, initial_const=para, max_iter=50, binary_search_steps=1,
                                     batch_size=5, learning_rate=0.1)

        (x_train, _), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        if model_type == 'NiN':
            x_train_mean = np.mean(x_train, axis=0)
            x_test -= x_train_mean
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

    elif model_type == 'cifar100':
        model = load_model(model_path)
        classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
        if attack_type == 'fgsm':
            attack = FastGradientMethod(estimator=classifier, eps=0.1, eps_step=0.01, minimal=True)
        elif attack_type == 'pgd':
            attack = ProjectedGradientDescent(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=100)
        elif attack_type == 'jsma':
            attack = SaliencyMapMethod(classifier=classifier, theta=0.1, gamma=0.01)
        elif attack_type == 'cw':
            attack = CarliniL2Method(classifier=classifier, max_iter=10, batch_size=50)
        (x_train, _), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        # x_train_mean = np.mean(x_train, axis=0)
        # x_test -= x_train_mean
        x_train, x_test = color_preprocessing(x_train, x_test)
        y_test = y_test.reshape(1, -1)[0]
        split = int(len(x_test) / 10)
        adv_num = 0
        total_num = 0
        for _ in range(10):
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

    elif model_type == 'IMDB':
        IMDB_attack(model_path, attack_type, results_path)

    elif model_type == 'QC':
        QC_attack(model_path, attack_type, results_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-model",
                        type=str,
                        )
    parser.add_argument("--results", "-results",
                        type=str,
                        )
    parser.add_argument("--model_type", "-model_type",
                        type=str,
                        )
    parser.add_argument("--attack", "-attack",
                        type=str,
                        )
    parser.add_argument("--para", "-para",
                        type=float,
                        )
    args = parser.parse_args()
    model_path = args.model
    results_path = args.results
    model_type = args.model_type
    attack_type = args.attack
    para = args.para
    art_attack(model_path, model_type, attack_type, para, results_path)
