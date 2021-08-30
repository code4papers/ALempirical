import tensorflow_model_optimization as tfmot
from keras.datasets import mnist, cifar10
from keras.models import load_model
import numpy as np
import tensorflow as tf
import csv
import argparse
from keras.optimizers import Adam


def pruning_train(model_path, data_type, save_path, results_path):
    model = load_model(model_path)
    if data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        batch_size = 128
        epochs = 5
        num_images = x_train.shape[0]
        end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
        test_acc = 0

        for _ in range(3):
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.10,
                                                                         final_sparsity=0.95,
                                                                         begin_step=0,
                                                                         end_step=end_step)
            }
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

            model_for_pruning = prune_low_magnitude(model, **pruning_params)
            model_for_pruning.compile(optimizer='adam',
                                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                      metrics=['accuracy'])

            model_for_pruning.summary()

            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
            ]

            his = model_for_pruning.fit(x_train, y_train,
                                        batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                                        callbacks=callbacks)
            model_for_pruning.save(save_path)
            test_acc += his.history['val_accuracy'][-1]
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, test_acc / 3])

        finally:
            csv_file.close()
    if data_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        batch_size = 256
        epochs = 50
        num_images = x_train.shape[0]
        end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
        test_acc = 0
        for _ in range(1):
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.05,
                                                                         final_sparsity=0.5,
                                                                         begin_step=50,
                                                                         end_step=end_step)
            }
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

            model_for_pruning = prune_low_magnitude(model, **pruning_params)
            model_for_pruning.compile(optimizer=Adam(lr=1e-3),
                                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                      metrics=['accuracy'])

            model_for_pruning.summary()
            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
            ]

            his = model_for_pruning.fit(x_train, y_train,
                                        batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                                        callbacks=callbacks)
            model_for_pruning.save(save_path)
            test_acc += his.history['val_accuracy'][-1]
        csv_file = open(results_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, test_acc / 3])

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
    parser.add_argument("--data", "-data",
                        type=str,
                        )
    parser.add_argument("--model_save", "-model_save",
                        type=str,
                        )

    args = parser.parse_args()
    model_path = args.model
    results_path = args.results
    data_type = args.data
    model_save_path = args.model_save

    pruning_train(model_path, data_type, model_save_path, results_path)
