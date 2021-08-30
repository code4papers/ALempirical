from test_selection import *
from ncoverage import NCoverage
import sys
sys.path.append('../../')

from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from tools import *
from NiN.NiNmodel import *
from VGG.VGG16models import *
from Lenet1.Lenet1_model import *
from Lenet_5 import *
from IMDB.IMDB_model import *
from QC.QC_model import *
from Yahoo.Yahoo_model import *


def test_selection_metrics(metrics, model, x, y, windows, model_type, layers):
    if model_type == 'lenet1' or model_type == 'lenet5' or model_type == 'NiN' or model_type == 'VGG16':
        num_classes = 10
    elif model_type == 'QC_lstm' or model_type == 'QC_gru':
        num_classes = 7
    elif model_type == 'Yahoo_lstm' or model_type == 'Yahoo_gru':
        num_classes = 10
    else:
        num_classes = 2
    # NC
    if metrics == 0:
        ncComputor = NCoverage(model, threshold=0.2)
        if model_type == 'NiN' or model_type == 'VGG16':
            (x_jointrain, y_jointrain), (x_remaining, y_remaining), nc_score = \
                getNeuroCover_spilt(model, (x, y), windows, method='NC', ncComputor=ncComputor)
        else:
            (x_jointrain, y_jointrain), (x_remaining, y_remaining), nc_score = \
                getNeuroCover(model, (x, y), windows, method='NC', ncComputor=ncComputor)
    # KMNC
    elif metrics == 1:
        ncComputor = NCoverage(model, threshold=0.2)
        # print(x.shape)
        inidata = x
        # print(inidata.shape)

        if model_type == 'NiN' or model_type == 'VGG16':
            ncComputor.initKMNCtable_split(inidata, 'nc_data/%s.p' % (model_type), read=False, save=False)
            (x_jointrain, y_jointrain), (x_remaining, y_remaining), nc_score = \
                getNeuroCover_spilt(model, (x, y), windows, method='KMNC', ncComputor=ncComputor)
        else:
            ncComputor.initKMNCtable_split(inidata, 'nc_data/%s.p' % (model_type), read=False, save=False)
            (x_jointrain, y_jointrain), (x_remaining, y_remaining), nc_score = \
                getNeuroCover_spilt(model, (x, y), windows, method='KMNC', ncComputor=ncComputor)
    # DSA
    elif metrics == 2:
        y_ref = np.squeeze(oneHot2Int(y))
        random_ref_idx = np.arange(len(x))
        np.random.shuffle(random_ref_idx)
        y = np.argmax(y, axis=1)
        xref = (x[random_ref_idx], y_ref[random_ref_idx], y[random_ref_idx])
        (x_jointrain, y_jointrain), (x_remaining, y_remaining), nc_score = \
            getSamplesDSA(model, (x, y), windows, xref=xref, layers=layers, num_classes=num_classes)
        y_jointrain = to_categorical(y_jointrain, num_classes)
        y_remaining = to_categorical(y_remaining, num_classes)
    # LSA
    elif metrics == 3:
        varthreshold = {'mlp': {'mnist': 1e-2, 'fashion_mnist': 1e-1},
                        'deepxplore': {'mnist': 1e-5, 'fashion_mnist': 1e-5},
                        'lenet': {'mnist': 1e-5, 'fashion_mnist': 1e-5},
                        'vgg': {'cifar10': 1e-1}, 'netinnet': {'cifar10': 1e-1},
                        }
        y_ref = np.squeeze(oneHot2Int(y))
        random_ref_idx = np.arange(len(x))
        np.random.shuffle(random_ref_idx)
        y = np.argmax(y, axis=1)
        xref = (x[random_ref_idx], y_ref[random_ref_idx], y[random_ref_idx])
        (x_jointrain, y_jointrain), (x_remaining, y_remaining), nc_score = \
            getSamplesLSA(model, (x, y), windows, xref=xref, layers=layers, num_classes=num_classes, varthreshold=varthreshold)
        y_jointrain = to_categorical(y_jointrain, num_classes)
        y_remaining = to_categorical(y_remaining, num_classes)
    # DeepGini
    elif metrics == 4:
        (x_jointrain, y_jointrain), (x_remaining, y_remaining), nc_score = DeepGini(model, (x, y), windows)

    # MCP
    elif metrics == 5:
        (x_jointrain, y_jointrain), (x_remaining, y_remaining), nc_score = MCP_selection(model, (x, y), windows)

    # Adversarial active learning
    else:
        (x_jointrain, y_jointrain), (x_remaining, y_remaining), nc_score = adversarial_selection(model, (x, y), windows, model_type)

    return (x_jointrain, y_jointrain), (x_remaining, y_remaining)


def test_selection_run(model_type, results_path, metric, model_save_path):
    if model_type == 'lenet5':
        steps = 20
        windows = 500
        epochs = 5
        model = Lenet5()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        dropout_model = Lenet5_dropout()
        dropout_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        target_data = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)
        layers = ['fc1']
        for _ in range(steps):
            (selected_data, selected_label), (remain_data, remain_label) = test_selection_metrics(metric, model, target_data, y_train, windows, model_type, layers)
            # print(remain_data.shape)
            # print(remain_label.shape)
            target_data = remain_data
            y_train = remain_label
            if _  == 0:
                training_data = selected_data
                training_label = selected_label
            else:
                training_data = np.concatenate((training_data, selected_data))
                training_label = np.concatenate((training_label, selected_label))
            train_len = len(training_data)
            print("training data len: {}".format(train_len))
            his = model.fit(training_data, training_label, batch_size=256, shuffle=True, epochs=epochs,
                            validation_data=(x_test, y_test), verbose=1)
            val_acc = his.history['val_accuracy'][-1]
            model.save(model_save_path)
            csv_file = open(results_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([train_len, val_acc])

            finally:
                csv_file.close()

        np.save("../../data/selected_data/Lenet5_SE/" + str(metric) + "_x_selected.npy", training_data)
        np.save("../../data/selected_data/Lenet5_SE/" + str(metric) + "_y_selected.npy", training_label)
        np.save("../../data/selected_data/Lenet5_SE/" + str(metric) + "_x_remain.npy", target_data)
        np.save("../../data/selected_data/Lenet5_SE/" + str(metric) + "_y_remain.npy", y_train)
        # model.save(model_save_path)

    elif model_type == 'lenet1':
        steps = 20
        windows = 500
        epochs = 5
        model = Lenet1()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        dropout_model = Lenet1_dropout()
        dropout_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        target_data = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)
        layers = ['conv2d_1']
        for _ in range(steps):
            (selected_data, selected_label), (remain_data, remain_label) = test_selection_metrics(metric, model,
                                                                                                  target_data, y_train,
                                                                                                  windows, model_type,
                                                                                                  layers)
            target_data = remain_data
            y_train = remain_label
            if _  == 0:
                training_data = selected_data
                training_label = selected_label
            else:
                training_data = np.concatenate((training_data, selected_data))
                training_label = np.concatenate((training_label, selected_label))
            train_len = len(training_data)
            print("training data len: {}".format(train_len))
            his = model.fit(training_data, training_label, batch_size=256, shuffle=True, epochs=epochs,
                            validation_data=(x_test, y_test), verbose=1)
            val_acc = his.history['val_accuracy'][-1]
            csv_file = open(results_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([train_len, val_acc])

            finally:
                csv_file.close()
        model.save(model_save_path)
    elif model_type == 'NiN':
        input_shape = (32, 32, 3)
        steps = 10
        windows = 2500
        epochs = 200
        # model = NIN(input_shape, 10)
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=Adam(lr=1e-3),
        #               metrics=['accuracy'])
        # model.save("/Users/qiang.hu/PycharmProjects/al_leak/new_models/RQ1/NiN/init.h5")
        model = load_model("/mnt/irisgpfs/users/qihu/pv_env/al_leak/new_models/RQ1/NiN/init.h5")

        dropout_model = NIN_all(input_shape, 10)
        dropout_model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=1e-3),
                      metrics=['accuracy'])

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Normalize data.
        target_data = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # If subtract pixel mean is enabled
        x_train_mean = np.mean(target_data, axis=0)
        target_data -= x_train_mean
        x_test -= x_train_mean
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        layers = ['conv2d_8']
        for _ in range(steps):
            (selected_data, selected_label), (remain_data, remain_label) = test_selection_metrics(metric, model,
                                                                                                  target_data, y_train,
                                                                                                  windows, model_type,
                                                                                                  layers)
            target_data = remain_data
            y_train = remain_label
            if _ == 0:
                training_data = selected_data
                training_label = selected_label
            else:
                training_data = np.concatenate((training_data, selected_data))
                training_label = np.concatenate((training_label, selected_label))
            train_len = len(training_data)
            print("training data len: {}".format(train_len))
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=1e-3),
                          metrics=['accuracy'])
            his = model.fit(training_data, training_label,
                                batch_size=128,
                                epochs=200,
                                validation_data=(x_test, y_test),
                                verbose=1)
            val_acc = his.history['val_accuracy'][-1]
            csv_file = open(results_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([train_len, val_acc])

            finally:
                csv_file.close()
        np.save("../../data/selected_data/NiN_SE/" + str(metric) + "_x_selected.npy", training_data)
        np.save("../../data/selected_data/NiN_SE/" + str(metric) + "_y_selected.npy", training_label)
        # np.save("../../data/selected_data/Lenet5_SE/" + str(metric) + "_x_remain.npy", target_data)
        # np.save("../../data/selected_data/Lenet5_SE/" + str(metric) + "_y_remain.npy", y_train)
        model.save(model_save_path)

    elif model_type == 'VGG16':
        steps = 20
        windows = 2500
        epochs = 200
        model = load_model("/mnt/irisgpfs/users/qihu/pv_env/al_leak/VGG/VGGmodels/init_VGG.h5")
        dropout_model = VGG16_clipped_dropout(input_shape=(32, 32, 3), rate=0.4,
                                              nb_classes=10)
        dropout_model.compile(loss='categorical_crossentropy',
                              optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),
                              metrics=['accuracy'])
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Normalize data.
        target_data = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # If subtract pixel mean is enabled

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        layers = ['dense_3']
        for _ in range(steps):
            if _ != 0:
                model = load_model(model_save_path + str(_ - 1) + ".h5")
            (selected_data, selected_label), (remain_data, remain_label) = test_selection_metrics(metric, model,
                                                                                                  target_data, y_train,
                                                                                                  windows, model_type,
                                                                                                  layers)
            target_data = remain_data
            y_train = remain_label
            if _ == 0:
                training_data = selected_data
                training_label = selected_label
            else:
                training_data = np.concatenate((training_data, selected_data))
                training_label = np.concatenate((training_label, selected_label))
            train_len = len(training_data)
            print("training data len: {}".format(train_len))
            bestmodelname = model_save_path + str(_) + ".h5"
            checkPoint = ModelCheckpoint(bestmodelname, monitor="val_accuracy", save_best_only=True, verbose=1)

            def lr_scheduler(epoch):
                initial_lrate = 1e-2
                drop = 0.9
                epochs_drop = 50.0
                lrate = initial_lrate * np.power(drop,
                                                 np.floor((1 + epoch) / epochs_drop))
                return lrate

            reduce_lr = callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
            train_datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
            train_datagen.fit(training_data)
            train_generator = train_datagen.flow(training_data, training_label, batch_size=128)
            nb_train_samples = training_data.shape[0] // 128
            his = model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples,
                epochs=200,
                validation_data=(x_test, y_test),
                validation_steps=10000 // 128,
                callbacks=[checkPoint, reduce_lr])

            val_acc = np.max(his.history['val_accuracy'])
            csv_file = open(results_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([train_len, val_acc])

            finally:
                csv_file.close()
            # if train_len >= 25000:
            #     break
    elif model_type == 'IMDB_lstm' or model_type == 'IMDB_gru':
        steps = 50
        windows = 500
        epochs = 5
        if model_type == 'IMDB_lstm':
            model = IMDB_LSTM_glove(emb_train=True)
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            dropout_model = IMDB_LSTM_glove_dropout()
            dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        else:
            model = IMDB_GRU_new()
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            dropout_model = IMDB_GRU_new_dropout()
            dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        maxlen = 200
        max_features = 20000
        (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
            num_words=max_features
        )
        print(len(x_train), "Training sequences")
        print(len(x_val), "Validation sequences")
        y_train = to_categorical(y_train, 2)
        y_val = to_categorical(y_val, 2)
        target_data = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
        x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
        layers = ['dense']
        for _ in range(steps):
            (selected_data, selected_label), (remain_data, remain_label) = test_selection_metrics(metric, model,
                                                                                                  target_data, y_train,
                                                                                                  windows, model_type,
                                                                                                  layers)
            target_data = remain_data
            y_train = remain_label
            if _  == 0:
                training_data = selected_data
                training_label = selected_label
            else:
                training_data = np.concatenate((training_data, selected_data))
                training_label = np.concatenate((training_label, selected_label))

            train_len = len(training_data)
            print("training data len: {}".format(train_len))
            best_model = model_save_path + str(_) + '.h5'
            check_point = ModelCheckpoint(best_model, monitor="val_accuracy", save_best_only=True, verbose=1)
            # his = model.fit(training_data, training_label, batch_size=32, shuffle=True, epochs=epochs,
            #                 validation_data=(x_val, y_val), verbose=1)
            his = model.fit(training_data, training_label, batch_size=32,
                            epochs=epochs, validation_data=(x_val, y_val),
                            callbacks=[check_point])

            val_acc = np.max(his.history['val_accuracy'])
            csv_file = open(results_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([train_len, val_acc])

            finally:
                csv_file.close()

    elif model_type == 'QC_lstm' or model_type == 'QC_gru':
        steps = 20
        windows = 1000
        epochs = 10
        data, label = get_QC_data("../../data/train_data_pytorch.csv", "../../data/test_data_pytorch.csv")
        train_indices = np.load("../../QC/data/training_indices.npy")
        test_indices = np.load("../../QC/data/test_indices.npy")
        target_data = data[train_indices]
        y_train = label[train_indices]
        x_test = data[test_indices]
        y_test = label[test_indices]
        y_train = to_categorical(y_train, 7)
        y_test = to_categorical(y_test, 7)
        layers = ['dense']
        if model_type == 'QC_lstm':
            model = QC_LSTM(emb_train=True)
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            dropout_model = QC_LSTM_dropout()
            dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        else:
            model = QC_GRU(emb_train=True)
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            dropout_model = QC_GRU_dropout()
            dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        for _ in range(steps):
            (selected_data, selected_label), (remain_data, remain_label) = test_selection_metrics(metric, model,
                                                                                                  target_data, y_train,
                                                                                                  windows, model_type,
                                                                                                  layers)
            target_data = remain_data
            y_train = remain_label
            if _ == 0:
                training_data = selected_data
                training_label = selected_label
            else:
                training_data = np.concatenate((training_data, selected_data))
                training_label = np.concatenate((training_label, selected_label))
            train_len = len(training_data)
            print("training data len: {}".format(train_len))
            check_point = ModelCheckpoint(model_save_path + str(_) + ".h5", monitor="val_accuracy", save_best_only=True, verbose=1)
            # his = model.fit(training_data, training_label, batch_size=32, shuffle=True, epochs=epochs,
            #                 validation_data=(x_val, y_val), verbose=1)
            his = model.fit(training_data, training_label, batch_size=128,
                            epochs=epochs, validation_data=(x_test, y_test),
                            callbacks=[check_point])

            val_acc = np.max(his.history['val_accuracy'])
            csv_file = open(results_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([train_len, val_acc])

            finally:
                csv_file.close()

    elif model_type == 'Yahoo_lstm' or model_type == 'Yahoo_gru':
        steps = 20
        windows = 178
        epochs = 16
        data, labels, texts = get_Yahoo_data()
        train_index = np.load("../../Yahoo/data/train_indices.npy")
        test_index = np.load("../../Yahoo/data/test_indices.npy")
        target_data = data[train_index]
        y_train = labels[train_index]
        x_test = data[test_index]
        y_test = labels[test_index]

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        layers = ['dense']
        if model_type == 'Yahoo_lstm':
            model = Yahoo_LSTM()
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            dropout_model = Yahoo_LSTM_dropout()
            dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        else:
            model = Yahoo_GRU()
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            dropout_model = Yahoo_GRU_dropout()
            dropout_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        for _ in range(steps):
            (selected_data, selected_label), (remain_data, remain_label) = test_selection_metrics(metric, model,
                                                                                                  target_data, y_train,
                                                                                                  windows, model_type,
                                                                                                  layers)
            target_data = remain_data
            y_train = remain_label
            if _ == 0:
                training_data = selected_data
                training_label = selected_label
            else:
                training_data = np.concatenate((training_data, selected_data))
                training_label = np.concatenate((training_label, selected_label))
            train_len = len(training_data)
            print("training data len: {}".format(train_len))
            best_model = model_save_path + str(_) + '.h5'
            check_point = ModelCheckpoint(best_model, monitor="val_accuracy", save_best_only=True, verbose=1)
            # his = model.fit(training_data, training_label, batch_size=32, shuffle=True, epochs=epochs,
            #                 validation_data=(x_val, y_val), verbose=1)
            his = model.fit(training_data, training_label, batch_size=32,
                            epochs=epochs, validation_data=(x_test, y_test),
                            callbacks=[check_point])

            val_acc = np.max(his.history['val_accuracy'])
            csv_file = open(results_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([train_len, val_acc])

            finally:
                csv_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", "-metric",
                        type=int,
                        )
    # 0-entropy 1-BALD 3-k_center 6-margin 7-entropy_dropout 8-margin_dropout
    parser.add_argument("--results", "-results",
                        type=str,
                        )
    parser.add_argument("--model", "-model",
                        type=str,
                        )
    parser.add_argument("--model_type", "-model_type",
                        type=str,
                        )
    args = parser.parse_args()
    metric = args.metric
    results_path = args.results
    model_save_path = args.model
    model_type = args.model_type
    test_selection_run(model_type, results_path, metric, model_save_path)
# y_ref = np.squeeze(oneHot2Int(y_train[:100]))
# random_ref_idx = np.arange(len(x_train[:100]))
# np.random.shuffle(random_ref_idx)
# y = np.argmax(y_train[:100], axis=1)
# xref = (x_train[random_ref_idx[:100]], y_ref[random_ref_idx[:100]], y[random_ref_idx[:100]])

# getSamplesDSA(model, (x_train[:100], y_train[:100]), 150, xref=xref, layers=['fc1'], num_classes=10)
# getSamplesLSA(model, (x_train[:100], y_train[:100]), 150, xref=xref, layers=['fc1'], num_classes=10, varthreshold=varthreshold)

# DeepGini(model, (x_train[:100], y_train[:100]), 50)
# MCP_selection(model, (x_train[:100], y_train[:100]), 50)
# adversarial_selection(model, (x_train[:100], y_train[:100]), 50)

