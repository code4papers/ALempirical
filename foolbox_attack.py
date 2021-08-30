import foolbox
from keras.models import load_model
from keras.datasets import mnist, cifar10
import numpy as np
import tensorflow as tf
import csv
import argparse


def foolbox_attack(model_path, model_type, attack_type, results_path, els, para=10, iter=10):
    model = load_model(model_path)
    if els == 10.0:
        els = None
    if model_type == 'lenet':
        if attack_type == 'fgsm':
            attack = foolbox.attacks.FGSM()
        elif attack_type == 'cw':
            attack = foolbox.attacks.L2CarliniWagnerAttack(steps=1000, stepsize=0.1, initial_const=para,
                                                           binary_search_steps=1)

        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype('float32') / 255
        split = int(len(x_test) / 100)
        adv_num = 0
        total_num = 0
        fm = foolbox.models.TensorFlowModel(model, bounds=(0, 1))
        for _ in range(100):
            print(_)
            x_part = x_test[split * _: split * (_ + 1)]
            y_part = y_test[split * _: split * (_ + 1)]
            ori_predictions = model.predict(x_part.reshape(-1, 28, 28, 1))
            ori_label = np.argmax(ori_predictions, axis=1)
            correct_classified = np.where(ori_label == y_part)[0]
            x_part = tf.convert_to_tensor(x_part[correct_classified])
            y_part = [_.astype(np.int32) for _ in y_part]
            y_part = np.asarray(y_part)
            y_part = tf.convert_to_tensor(y_part[correct_classified])
            raw, clipped, is_adv = attack(fm, x_part, y_part, epsilons=els)
            is_adv = is_adv.numpy()
            is_adv = is_adv.reshape(1, -1)[0]
            # print(is_adv)
            adv_num += len(np.where(is_adv == True)[0])
            total_num += len(is_adv)

        robust_accuracy = adv_num / total_num
        print(robust_accuracy)
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, attack_type, robust_accuracy])

        finally:
            csv_file.close()
    elif model_type == 'NiN' or model_type == 'VGG16':
        if attack_type == 'fgsm':
            attack = foolbox.attacks.FGSM()
        if attack_type == 'cw':
            attack = foolbox.attacks.L2CarliniWagnerAttack(steps=1000, stepsize=0.1, initial_const=para,
                                                           binary_search_steps=1)


        (x_train, _), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        if model_type == 'NiN':
            x_train_mean = np.mean(x_train, axis=0)
            x_test -= x_train_mean
        y_test = y_test.reshape(1, -1)[0]
        # print(y_test)

        split = int(len(x_test) / 100)
        fm = foolbox.models.TensorFlowModel(model, bounds=(0, 1))
        adv_num = 0
        total_num = 0
        for _ in range(100):
            x_part = x_test[split * _: split * (_ + 1)]
            y_part = y_test[split * _: split * (_ + 1)]
            # print(y_part)
            ori_predictions = model.predict(x_part)
            ori_label = np.argmax(ori_predictions, axis=1)
            # print(ori_label)

            correct_classified = np.where(ori_label == y_part)[0]
            # print(correct_classified)
            x_part = tf.convert_to_tensor(x_part[correct_classified])
            y_part = [_.astype(np.int32) for _ in y_part]
            y_part = np.asarray(y_part)
            # print(y_part)
            y_part = tf.convert_to_tensor(y_part[correct_classified])

            raw, clipped, is_adv = attack(fm, x_part, y_part, epsilons=els)
            is_adv = is_adv.numpy()
            is_adv = is_adv.reshape(1, -1)[0]
            # print(model.predict(clipped))
            # print(is_adv)
            adv_num += len(np.where(is_adv==True)[0])
            total_num += len(is_adv)
        robust_accuracy = adv_num / total_num
        print(robust_accuracy)
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, attack_type, robust_accuracy])

        finally:
            csv_file.close()


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
                        default=10
                        )
    parser.add_argument("--els", "-els",
                        type=float,
                        default=10.0
                        )
    parser.add_argument("--iter", "-iter",
                        type=int,
                        default=10
                        )
    args = parser.parse_args()
    model_path = args.model
    results_path = args.results
    model_type = args.model_type
    attack_type = args.attack
    para = args.para
    els = args.els
    iter = args.iter
    foolbox_attack(model_path, model_type, attack_type, results_path, els, para, iter)
