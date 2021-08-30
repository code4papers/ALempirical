from scipy.stats import entropy
import numpy as np
from scipy import stats
from utils import *
from sklearn.cluster import KMeans
from datetime import datetime
from keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.python.keras.backend import eager_learning_phase_scope
import pickle
from coreset import *
import tensorflow as tf
# import tensorflow.keras.backend as K


def entropy_selection(model, target_data, target_label, select_size):
    print("Prepare...")
    prediction = model.predict(target_data)
    entropy_list = entropy(prediction, base=2, axis=1)
    # print(entropy_list)
    # print(prediction[0])
    # print(len(entropy_list))
    sorted_index = np.argsort(entropy_list)
    select_index = sorted_index[-select_size:]
    # print(select_index)
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def BALD_selection(dropout_model, target_data, target_label, select_size):
    BALD_list = []
    mode_list = []
    data_len = len(target_data)
    print("Prepare...")
    for _ in range(20):
        prediction = np.argmax(dropout_model.predict(target_data), axis=1)
        BALD_list.append(prediction)
    BALD_list = np.asarray(BALD_list)
    for _ in range(data_len):
        mode_num = stats.mode(BALD_list[:, _:(_ + 1), ].reshape(-1,))[1][0]

        mode_list.append(1 - mode_num / 50)

    sorted_index = np.argsort(mode_list)
    select_index = sorted_index[-(select_size):]
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    print("target len, ", len(target_data))
    print("select size, ", select_size)
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def Kmeans_selection(ori_model, target_data, target_label, select_size):
    embedding = get_embedding(ori_model, target_data, 5)
    embedding = np.asarray(embedding)
    print("k-means fit...")
    cluster_learner = KMeans(n_clusters=select_size)
    cluster_learner.fit(embedding)
    print("fit over...")
    cluster_idxs = cluster_learner.predict(embedding)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embedding - centers) ** 2
    dis = dis.sum(axis=1)
    select_index = np.array([np.arange(embedding.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(select_size)])
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    print("target len, ", len(target_data))
    print("select size, ", select_size)
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def k_center_greedy_selection(ori_model, target_data, target_label, select_size, lb):
    # split = int(len(target_data) / 3)
    embedding = None
    data_len = len(target_data)
    split = int(data_len / 100)
    for i in range(100):
        # print(i)
        embedding_1 = get_embedding(ori_model, target_data[i * split: (i + 1) * split], -2)
        embedding_1 = np.asarray(embedding_1)
        if i == 0:
            embedding = embedding_1
        # print(emb_single.shape)
        else:
            embedding = np.concatenate((embedding, embedding_1))
    # embedding = get_embedding(ori_model, target_data, -2)

    print('calculate distance matrix')
    t_start = datetime.now()
    dist_mat = np.matmul(embedding, embedding.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(target_label), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    #
    dist_mat = np.sqrt(dist_mat)
    print("hhahahahahahaah")

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


def least_confidence(ori_model, target_data, target_label, select_size):
    prediction = ori_model.predict(target_data)
    max_pre = prediction.max(1)
    sorted_index = np.argsort(max_pre)
    select_index = sorted_index[-select_size:]
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def margin_selection(ori_model, target_data, target_label, select_size):
    prediction = ori_model.predict(target_data)
    prediction_sorted = np.sort(prediction)

    margin_list = prediction_sorted[:, -1] - prediction_sorted[:, -2]
    sorted_index = np.argsort(margin_list)
    select_index = sorted_index[: select_size]
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def entropy_dropout_selection(dropout_model, target_data, target_label, select_size):
    print("Prepare...")
    prediction = np.asarray(dropout_model.predict(target_data))
    for _ in range(1, 20):
        prediction += np.asarray(dropout_model.predict(target_data))
    prediction /= 20
    entropy_list = entropy(prediction, base=2, axis=1)
    sorted_index = np.argsort(entropy_list)
    select_index = sorted_index[-select_size:]
    print(select_index)
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def margin_dropout_selection(dropout_model, target_data, target_label, select_size):
    print("Prepare...")
    prediction = np.asarray(dropout_model.predict(target_data))
    for _ in range(1, 20):
        prediction += np.asarray(dropout_model.predict(target_data))
    prediction /= 20
    prediction_sorted = np.sort(prediction)
    margin_list = prediction_sorted[:, -1] - prediction_sorted[:, -2]
    sorted_index = np.argsort(margin_list)
    select_index = sorted_index[: select_size]
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def least_confidence_dropout_selection(dropout_model, target_data, target_label, select_size):
    print("Prepare...")
    prediction = np.asarray(dropout_model.predict(target_data))
    for _ in range(1, 20):
        prediction += np.asarray(dropout_model.predict(target_data))
    prediction /= 20
    max_pre = prediction.max(1)
    sorted_index = np.argsort(max_pre)
    select_index = sorted_index[-select_size:]
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


# useless
def coreset_selection_2(ori_model, trained_data, trained_label, target_data, target_label, select_size):
    trained_embedding = get_embedding(ori_model, trained_data, 5)
    trained_embedding = np.asarray(trained_embedding)
    target_embedding = get_embedding(ori_model, target_data, 5)
    target_embedding = np.asarray(target_embedding)
    all_embedding = trained_embedding + target_embedding
    labeled_indices = np.arange(0, len(trained_embedding))
    coreset = Coreset_Greedy(all_embedding)
    new_batch, max_distance = coreset.sample(labeled_indices, select_size)
    select_index = [i - len(trained_embedding) for i in new_batch]
    print(new_batch)
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    trained_data = np.concatenate((trained_data, selected_data), axis=0)
    trained_label = np.concatenate((trained_label, selected_label), axis=0)
    return selected_data, selected_label, remain_data, remain_label, trained_data, trained_label


def coreset_selection(ori_model, target_data, target_label, select_size, lb):
    embedding = get_embedding(ori_model, target_data, 5)
    embedding = np.asarray(embedding)
    print('calculate distance matrix')
    t_start = datetime.now()
    dist_mat = np.matmul(embedding, embedding.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(target_data), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)
    print(datetime.now() - t_start)

    print('calculate greedy solution')
    t_start = datetime.now()
    lb_flag = lb.copy()
    mat = dist_mat[~lb_flag, :][:, lb_flag]
    for i in range(select_size):
        if i % 10 == 0:
            print('greedy solution {}/{}'.format(i, len(target_data)))
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(len(target_label))[~lb_flag][q_idx_]
        lb_flag[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

    print(datetime.now() - t_start)
    opt = mat.min(axis=1).max()

    xx, yy = np.where(dist_mat <= opt)
    dd = dist_mat[xx, yy]

    lb_flag_ = lb.copy()
    subset = np.where(lb_flag_ == True)[0].tolist()
    SEED = 5
    pickle.dump((xx.tolist(), yy.tolist(), dd.tolist(), subset, float(opt), select_size, len(target_data)),
                open('mip{}.pkl'.format(SEED), 'wb'), 2)
    r_name = 'mip{}.pkl'.format(SEED)
    w_name = 'sols.pkl'.format(SEED)
    get_sols(r_name, w_name)
    sols = pickle.load(open('sols{}.pkl'.format(SEED), 'rb'))

    if sols is None:
        q_idxs = lb_flag
    else:
        lb_flag_[sols] = True
        q_idxs = lb_flag_
    print('sum q_idxs = {}'.format(q_idxs.sum()))
    select_index = np.arange(len(target_data))[(lb ^ q_idxs)]
    lb[select_index] = True
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    print("target len, ", len(target_data))
    print("select size, ", select_size)
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label, lb


def EGL_compute(model, unlabeled, n_classes):
    model(tf.keras.Input((model.input_shape[-3], model.input_shape[-2], model.input_shape[-1])))
    input_placeholder = K.placeholder(model.get_input_shape_at(0))
    output_placeholder = K.placeholder(model.get_output_shape_at(0))
    predict = model.call(input_placeholder)
    loss = K.mean(categorical_crossentropy(output_placeholder, predict))
    weights = [tensor for tensor in model.trainable_weights]
    gradient = model.optimizer.get_gradients(loss, weights)
    gradient_flat = [K.flatten(x) for x in gradient]
    gradient_flat = K.concatenate(gradient_flat)
    gradient_length = tf.keras.backend.sum(K.square(gradient_flat))
    get_gradient_length = K.function([input_placeholder, output_placeholder], [gradient_length])
    unlabeled_predictions = model.predict(unlabeled)
    egls = np.zeros(unlabeled.shape[0])
    for i in range(n_classes):
        calculated_so_far = 0
        while calculated_so_far < unlabeled_predictions.shape[0]:
            if calculated_so_far + 100 >= unlabeled_predictions.shape[0]:
                next = unlabeled_predictions.shape[0] - calculated_so_far
            else:
                next = 100

            labels = np.zeros((next, n_classes))
            labels[:, i] = 1
            # with eager_learning_phase_scope(value=0):
            grads = get_gradient_length([unlabeled[calculated_so_far:calculated_so_far + next, :], labels])[0]
            grads *= unlabeled_predictions[calculated_so_far:calculated_so_far + next, i]
            egls[calculated_so_far:calculated_so_far + next] += grads

            calculated_so_far += next

    return egls


def EGL_selection(model, target_data, target_label, select_size):
    n_classes = 10
    egls = EGL_compute(model, target_data, n_classes)
    select_index = np.argsort(egls)[-select_size:]

    # select_index = np.argpartition(egls, -select_size)[-select_size:]
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label


def EGL_selection_index(model, target_data, target_label, select_size):
    n_classes = 10
    egls = EGL_compute(model, target_data, n_classes)
    select_index = np.argsort(egls)[-select_size:]

    # select_index = np.argpartition(egls, -select_size)[-select_size:]
    # selected_data = target_data[select_index]
    # selected_label = target_label[select_index]
    # remain_data = np.delete(target_data, select_index, axis=0)
    # remain_label = np.delete(target_label, select_index, axis=0)
    return select_index


def random_selection(model, target_data, target_label, select_size):
    select_index = np.random.randint(0, len(target_data), select_size)
    selected_data = target_data[select_index]
    selected_label = target_label[select_index]
    remain_data = np.delete(target_data, select_index, axis=0)
    remain_label = np.delete(target_label, select_index, axis=0)
    return selected_data, selected_label, remain_data, remain_label





