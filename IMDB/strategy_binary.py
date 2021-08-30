from keras.models import load_model
import keras
import numpy as np
import keras.backend as K
from progressbar import *
from keras.losses import binary_crossentropy, categorical_crossentropy
from scipy.stats import entropy
from utils import *
import tensorflow as tf
from tensorflow.python.keras.backend import eager_learning_phase_scope


def build_dropout_model(ori_model, dropout_model):
    model_weights = ori_model.get_weights()
    dropout_model.set_weights(model_weights)
    return dropout_model


def BALD_selection(dropout_model, target_data, target_label, select_size):
    BALD_list = []
    mode_list = []
    # calculate sensitivity of each data
    data_len = len(target_data)
    print("Start collecting diversity matrix")
    for _ in range(20):
        prediction = dropout_model.predict(target_data)
        BALD_list.append(prediction)
    BALD_list = np.asarray(BALD_list)
    for _ in range(data_len):
        BALD_single = BALD_list[:, _:(_ + 1), ]
        left_ = len(np.where(BALD_single > 0.5)[0])
        count_num = left_ if left_ > 10 else (20 - left_)
        mode_list.append(1 - count_num / 20)

    sorted_index = np.argsort(mode_list)
    select_index = sorted_index[-(select_size):]
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    print("target len, ", len(target_data))
    print("select size, ", select_size)
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def compute_egls(model, unlabeled, n_classes):
    # print(model.input_shape)
    model(tf.keras.Input((200)))
    # create a function for computing the gradient length:
    input_placeholder = K.placeholder(model.get_input_shape_at(0))
    print(input_placeholder)
    output_placeholder = K.placeholder(model.get_output_shape_at(0))
    print(output_placeholder)
    predict = model.call(input_placeholder)
    loss = K.mean(binary_crossentropy(output_placeholder, predict))
    weights = [tensor for tensor in model.trainable_weights]
    gradient = model.optimizer.get_gradients(loss, weights)
    gradient_flat = [K.flatten(x) for x in gradient]
    gradient_flat = K.concatenate(gradient_flat)
    # gradient_flat = tf.concat(gradient_flat, axis=-1)
    gradient_length = tf.keras.backend.sum(K.square(gradient_flat))
    # gradient_length = tf.keras.backend.eval(gradient_length)
    print("#####################################")
    print(type(gradient_length))
    print("#####################################")
    get_gradient_length = K.function([input_placeholder, output_placeholder],
                                     [tf.keras.backend.eval(gradient_length)])

    # calculate the expected gradient length of the unlabeled set (iteratively, to avoid memory issues):
    unlabeled_predictions = model.predict(unlabeled)

    egls = np.zeros(unlabeled.shape[0])

    for i in range(n_classes):
        calculated_so_far = 0
        # print("################, ", unlabeled_predictions.shape[0])
        print("start collecting label : {}".format(i))
        p_bar = ProgressBar().start()
        data_len = unlabeled_predictions.shape[0]
        print("hahahah, ", data_len)
        while calculated_so_far < unlabeled_predictions.shape[0]:
            p_bar.update(int((calculated_so_far / data_len) * 100))
            if calculated_so_far + 100 >= unlabeled_predictions.shape[0]:
                next = unlabeled_predictions.shape[0] - calculated_so_far
            else:
                next = 100
            labels = np.zeros((next, 1))
            if i == 0:
                labels[:i] = 0
            else:
                labels[:i] = 1
            with eager_learning_phase_scope(value=0):
                grads = get_gradient_length([unlabeled[calculated_so_far:calculated_so_far + next, :], labels])[0]
            grads *= unlabeled_predictions[calculated_so_far:calculated_so_far + next, 0]
            egls[calculated_so_far:calculated_so_far + next] += grads

            calculated_so_far += next
        p_bar.finish()

    return egls


def EGL_selection(model, target_data, target_label, select_size):
    n_classes = 2
    egls = compute_egls(model, target_data, n_classes)
    select_index = np.argsort(egls)[-select_size:]

    # select_index = np.argpartition(egls, -select_size)[-select_size:]
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def random_selection(target_data, target_label, select_size):
    select_index = np.random.choice(len(target_data), select_size, replace=False)
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def entropy_selection(model, target_data, target_label, select_size):
    predictions = model.predict(target_data)
    entropy_list = [entropy([predictions[_], 1 - predictions[_]], base=2)[0] for _ in range(len(target_data))]

    sorted_index = np.argsort(entropy_list)
    select_index = sorted_index[-select_size:]
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def entropy_dropout_selection(dropout_model, target_data, target_label, select_size):
    print("Prepare...")
    prediction = np.asarray(dropout_model.predict(target_data))
    entropy_list = np.array([entropy([prediction[_], 1 - prediction[_]], base=2)[0] for _ in range(len(target_data))])
    for _ in range(1, 20):
        prediction = np.asarray(dropout_model.predict(target_data))
        entropy_list += [entropy([prediction[_], 1 - prediction[_]], base=2)[0] for _ in range(len(target_data))]
    entropy_list /= 20
    sorted_index = np.argsort(entropy_list)
    select_index = sorted_index[-select_size:]
    print(select_index)
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def k_center_greedy_selection(ori_model, target_data, target_label, select_size, lb):
    # split = int(len(target_data) / 3)
    embedding = None
    data_len = len(target_data)
    split = int(data_len / 50)
    for i in range(50):
        embedding_1 = get_embedding(ori_model, target_data[i * split: (i + 1) * split], -2)
        embedding_1 = np.asarray(embedding_1)
        if i == 0:
            embedding = embedding_1
        # print(emb_single.shape)
        else:
            embedding = np.concatenate((embedding, embedding_1))

    print('calculate distance matrix')
    t_start = datetime.now()
    dist_mat = np.matmul(embedding, embedding.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(target_label), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)
    print(datetime.now() - t_start)
    lb_flag = lb.copy()
    mat = dist_mat[~lb_flag, :][:, lb_flag]
    for i in range(select_size):
        if i % 10 == 0:
            print('greedy solution {}/{}'.format(i, select_size))
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(len(target_label))[~lb_flag][q_idx_]
        lb_flag[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)
    select_index = np.arange(len(target_label))[(lb ^ lb_flag)]
    lb[select_index] = True
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    print("target len, ", len(target_data))
    print("select size, ", select_size)
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label, lb


def load_imdb_data(max_features):
    max_features = max_features  # Only consider the top 20k words

    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
        num_words=max_features
    )
    return (x_train, y_train), (x_val, y_val)

