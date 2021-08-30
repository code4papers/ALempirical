import numpy as np
#np.random.seed(698686)
import tools as utils
import sa as sa
from MCP import *
import foolbox
import tensorflow as tf

def computeVariancescore(model, data, drop_rep=50, num_class=10):
    '''
    :param model:
    :param data:
    :param drop_rep:
    :return:
    '''
    # X, Y,num_repeat, num_class, model
    (x, y, nb_class) = data
    (result, label, _, _, _, _) = \
            utils.predict(x, y, drop_rep, num_class, model)
    # Sort variance
    var_all_class = np.var(result, axis=0)
    var_mean_all_class = np.mean(var_all_class, axis=1)
    del model, data, x, y
    return var_mean_all_class


def getSamplesByVar(model, remainingData, num, drop_rep=50, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param num:
    :param drop_rep:
    :param drop_rate:
    :param dataset:
    :return:
    '''
    #X, Y,num_repeat, num_class, model
    (x,y) = remainingData

    (result, label, counter, p_rlabel, variance, means) = \
        utils.predict(x, y, drop_rep, kwargs["num_class"], model)
    #Sort variance
    var_all_class = np.var(result, axis=0)
    var_mean_all_class = np.mean(var_all_class, axis=1)
    p, _ = utils.prob_mean(result)

    if kwargs['method'] == 'var':
        ind = np.argsort(var_mean_all_class)[::-1]
        (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
            selectData(x[ind], y[ind], kwargs["num_each"], kwargs['num_class'])
    elif kwargs['method']== 'varW':
        ormodel = kwargs['orginalModel']
        py = ormodel.predict(x)
        pl = np.argmax(py, axis=1)
        pp = np.squeeze(py[np.arange(len(py)), pl])
        var_mean_all_class = np.squeeze(var_mean_all_class)
        var_mean_all_class = np.divide(var_mean_all_class, pp)
        ind = np.argsort(var_mean_all_class)[::-1]
        (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
            selectData(x[ind], y[ind], kwargs["num_each"], kwargs['num_class'])
    else :
        # select data by two dimesions output probability and the variance
        nb_bins = 50
        dic, score, p, vbins, pbins= utils.compute2DHistGroup(var_mean_all_class, p, nb_bins)
        res = []
        for i in range(nb_bins)[::-1]:
            vidx, pidx = i, nb_bins-1-i
            for n in range(pidx+1):
                res.extend(dic[nb_bins-1-n][pidx])
            for h in range(pidx):
                res.extend(dic[nb_bins-1-pidx][h])
            if len(res)>=num:
                break
        res = np.asarray(res)
        idx = res[:num]
        idx_left = np.ones(len(x), dtype=bool)
        idx_left[idx] = False
        x_jointrain, y_jointrain = x[idx], y[idx]
        x_remaining, y_remaining = x[idx_left], y[idx_left]

    del result
    del label
    del counter
    del p_rlabel
    del variance
    del means,model
    return (x_jointrain, y_jointrain),(x_remaining, y_remaining),var_mean_all_class


def getSamplesRandom(remainingData, num, **kwargs):
    (x, y) = remainingData
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
        selectData(x[idx], y[idx], kwargs['num_each'], kwargs['num_class'])
    del x
    del y, remainingData
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining), None


def computeDSAscore(model, remainingData, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param kwargs:
    :return:
    '''

    (x_reference, y_reference, y_pre) = kwargs["xref"]
    print(x_reference.shape)
    print(y_reference.shape)
    print(y_pre.shape)
    (x, y) = remainingData

    dsascores = sa.fetch_dsa(model, x_reference, x, "candidates", kwargs["layers"], num_classes=kwargs["num_classes"], var_threshold=1e-5,
                             is_classification=True)
    dsascores = np.asarray(dsascores)

    del x_reference
    del y_reference
    del y_pre
    del x
    del y
    del remainingData, model
    return dsascores


def getSamplesDSA(model, remainingData, num, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param num:
    :param kwargs:
    :return:
    '''
    (x, y) = remainingData
    dsascores = computeDSAscore(model, remainingData, **kwargs)#np.zeros(len(x))
    idx = np.argsort(dsascores)[::-1]
    print(idx)
    select_index = idx[:num]
    x_jointrain = x[select_index]
    y_jointrain = y[select_index]
    x_remaining = np.delete(x, select_index, axis=0)
    y_remaining = np.delete(y, select_index, axis=0)
    # (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
    #     selectData(x[idx], y[idx],  kwargs['num_each'], kwargs['num_class'])

    del x
    del y
    del remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining), dsascores


def computeLSAscore(model, remainingData, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param kwargs:
    :return:
    '''

    (x_reference, y_reference, y_pre) = kwargs["xref"]
    (x, _) = remainingData

    lsascores = sa.fetch_dsa(model, x_reference, x, "candidates", kwargs["layers"], num_classes=kwargs["num_classes"],
                             var_threshold=kwargs['varthreshold'],
                             is_classification=True)

    del  x_reference,  y_reference  , y_pre, x, model
    return lsascores


def getSamplesLSA(model, remainingData, num, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param num:
    :param kwargs:
    :return:
    '''
    (x, y) = remainingData
    lsascores = computeLSAscore(model, remainingData, **kwargs)

    idx = np.argsort(lsascores)[::-1]
    select_index = idx[:num]
    x_jointrain = x[select_index]
    y_jointrain = y[select_index]
    x_remaining = np.delete(x, select_index, axis=0)
    y_remaining = np.delete(y, select_index, axis=0)
    # (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
    #     selectData(x[idx], y[idx],  kwargs['num_each'], kwargs['num_class'])
    del x, y, remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining),lsascores


def computeSilhouttescore(model, remainingData, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param kwargs:
    :return:
    '''

    (x_reference, y_reference, y_pre) = kwargs["xref"]
    (x, _) = remainingData

    sihoutete = sa.fetch_sihoutete(model, x_reference, x, "candidates", kwargs["layers"], num_classes=10, var_threshold=1e-5,
                                   is_classification=True)
    del  x_reference,  y_reference  , y_pre, x, model
    return sihoutete


def getSamplesSilhoutte(model, remainingData, num, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param num:
    :param kwargs:
    :return:
    '''
    (x, y) = remainingData
    silhoutte = computeSilhouttescore(model, remainingData, **kwargs)

    idx = np.argsort(silhoutte)
    (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
        selectData(x[idx], y[idx],  kwargs['num_each'], kwargs['num_class'])
    del x, y, remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining),silhoutte


def selectData(x, y, num_each, nb_classes):
    newy = ()
    newx = ()
    leftx = ()
    lefty = ()

    def f(x, y):
        x  = np.concatenate(x, axis=0)
        #print(x.shape)
        y = np.concatenate(y, axis=0)
        #print(y.shape)
        a = np.arange(len(x))
        np.random.shuffle(a)
        return (x[a], y[a])

    for i in num_each:
        if num_each[i] == 0:
            continue
        idx = (y == i)
        newy += (y[idx][:num_each[i]], )
        newx += (x[idx][:num_each[i]], )
        leftx += (x[idx][num_each[i]:],)
        lefty += (y[idx][num_each[i]:],)
    return f(newx, newy), f(leftx, lefty)


# #Test
def computeKLScore(model, remainingData, drop_rep=50, **kwargs):
    '''

        :param model:
        :param remainingData:
        :param num:
        :param drop_rep:
        :param drop_rate:
        :param dataset:
        :return:
        '''
    # X, Y,num_repeat, num_class, model
    (x, y) = remainingData
    (result, label, _, _, _, _) = \
        utils.predict(x, y, drop_rep, kwargs["num_class"], model)

    # Sort divergence
    kl, var_hist = utils.computeKL(result, kwargs["num_class"], label)
    del result, label, model
    return kl, var_hist


def getKLDiverge(model, remainingData, num, drop_rep=50, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param num:
    :param drop_rep:
    :param drop_rate:
    :param dataset:
    :return:
    '''
    #X, Y,num_repeat, num_class, model
    (x,y) = remainingData

    (result, label, counter, p_rlabel, variance, means) = \
        utils.predict(x, y, drop_rep, kwargs["num_class"], model)
    #Sort divergence
    kl, var_hist = utils.computeKL(result, kwargs["num_class"], label)
    if kwargs["method"]=='KL':
        ind = np.argsort(kl)
        (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
            selectData(x[ind], y[ind], kwargs['num_each'], kwargs['num_class'])
    else:
        nb_bins = 50
        p,_ = utils.prob_mean(result)
        dic, score, p, vbins, pbins = utils.compute2DHistGroup(kl, p, nb_bins)
        res = []
        for i in range(nb_bins):
            vidx, pidx = i, i
            for n in range(pidx + 1):
                res.extend(dic[n][pidx])
            for h in range(vidx):
                res.extend(dic[vidx][h])
            if len(res) >= num:
                break
        res = np.asarray(res)
        idx = res[:num]
        idx_left = np.ones(len(x), dtype=bool)
        idx_left[idx] = False
        x_jointrain, y_jointrain = x[idx], y[idx]
        x_remaining, y_remaining = x[idx_left], y[idx_left]

    del result, label, counter, p_rlabel, variance, means,model
    return (x_jointrain, y_jointrain),(x_remaining, y_remaining),kl


def getLabelHist(model, remainingData, num, drop_rep=50, **kwargs):
    '''

    :param model:
    :param remainingData:
    :param num:
    :param drop_rep:
    :param drop_rate:
    :param dataset:
    :return:
    '''
    #X, Y,num_repeat, num_class, model
    (x,y) = remainingData

    (result, label, counter, p_rlabel, variance, means) = \
        utils.predict(x, y, drop_rep, kwargs["num_class"], model)
    #Sort divergence
    _, var_hist = utils.computeKL(result, kwargs["num_class"], label)
    ind = np.argsort(var_hist)
    (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
        selectData(x[ind], y[ind], kwargs['num_each'], kwargs['num_class'])
    del result, label, counter, p_rlabel, variance, means,model
    return (x_jointrain, y_jointrain),(x_remaining, y_remaining),var_hist


def DeepGini(model, remainingData, num, **kwargs):
    """
    DeepGini
    """
    (x, y) = remainingData
    pred_test_prob = model.predict(x)
    metrics = np.sum(pred_test_prob ** 2, axis=1)
    rank_lst = np.argsort(metrics)
    select_index = rank_lst[:num]
    print(select_index)
    x_jointrain = x[select_index]
    y_jointrain = y[select_index]
    x_remaining = np.delete(x, select_index, axis=0)
    y_remaining = np.delete(y, select_index, axis=0)
    del x, y, remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining), None


def MCP_selection(model, remainingData, num, **kwargs):
    (x, y) = remainingData
    rank_list = select_my_optimize(model, num, x, y)
    select_index = rank_list[:num]
    print(select_index)
    x_jointrain = x[select_index]
    y_jointrain = y[select_index]
    x_remaining = np.delete(x, select_index, axis=0)
    y_remaining = np.delete(y, select_index, axis=0)
    del x, y, remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining), None


def adversarial_selection(model, remainingData, num, model_type, **kwargs):
    (x, y) = remainingData
    # x = x[:200]
    # y = y[:200]
    fm = foolbox.models.TensorFlowModel(model, bounds=(0, 1))
    true_predictions = np.argmax(model.predict(x, batch_size=256), axis=1)
    adversarial_predictions = np.copy(true_predictions)

    while np.sum(true_predictions != adversarial_predictions) < num:
        attack_step = 10
        if model_type == 'NiN':
            attack = foolbox.attacks.FGSM()
        else:
            attack = foolbox.attacks.L2DeepFoolAttack(candidates=10, steps=attack_step)
        adversarial_images = np.zeros(x.shape)
        for i in range(0, x.shape[0], 100):
            print("At {i} out of {n}".format(i=i, n=x.shape[0]))
            if i + 100 > x.shape[0]:
                y_part = true_predictions[i:]
                y_part = [_.astype(np.int32) for _ in y_part]
                y_part = np.asarray(y_part)
                y_part = tf.convert_to_tensor(y_part)
                raw, clipped, is_adv = attack(fm, tf.convert_to_tensor(x[i:]), y_part, epsilons=0.03)
                adversarial_images[i:] = clipped
            else:
                y_part = true_predictions[i:i + 100]
                print(y_part)
                y_part = [_.astype(np.int32) for _ in y_part]
                y_part = np.asarray(y_part)
                y_part = tf.convert_to_tensor(y_part)

                raw, clipped, is_adv = attack(fm, tf.convert_to_tensor(x[i:i + 100]), y_part, epsilons=1)
                print(is_adv)
                adversarial_images[i:i + 100] = clipped
        pertubations = adversarial_images - x
        norms = np.linalg.norm(np.reshape(pertubations, (x.shape[0], -1)), axis=1)
        adversarial_predictions = np.argmax(model.predict(adversarial_images, batch_size=256), axis=1)
        norms[true_predictions == adversarial_predictions] = np.inf
        attack_step *= 2
    select_index = np.argpartition(norms, num)[:num]
    x_jointrain = x[select_index]
    y_jointrain = y[select_index]
    x_remaining = np.delete(x, select_index, axis=0)
    y_remaining = np.delete(y, select_index, axis=0)
    del x, y, remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining), None


def getNeuroCover(model, remainingData, num, **kwargs):
    (x, y) = remainingData
    ncComputor = kwargs['ncComputor']
    nc_score = None
    if kwargs['method'] == 'NC':
        nc_score = ncComputor.batch_nc(x)
    if kwargs['method'] == 'KMNC':
        (nc_score,_,_) = ncComputor.batch_kmnc(x)
    if kwargs['method'] == 'BNC':
        (_,nc_score, _) = ncComputor.batch_kmnc(x)
    if kwargs['method'] == 'SANC':
        (_,_, nc_score) = ncComputor.batch_kmnc(x)

    if kwargs['method'] == 'DiffNC':
        nc_score = ncComputor.batch_diffScore(x)

    # print(nc_score)
    assert nc_score is not None
    idx = np.argsort(nc_score)[::-1]
    select_index = idx[:num]
    x_jointrain = x[select_index]
    y_jointrain = y[select_index]
    x_remaining = np.delete(x, select_index, axis=0)
    y_remaining = np.delete(y, select_index, axis=0)
    # print(nc_score[idx[0]])
    # if np.any(nc_score==0):
    #     non_zero = np.nonzero(nc_score)[0]
    #     print(non_zero)
    #     usedIdx = idx[:non_zero.size]
    #     leftIdx = idx[non_zero.size:]
    #     np.random.shuffle(leftIdx)
    #     idx = np.concatenate([usedIdx, leftIdx])
    #
    # (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
    #     selectData(x[idx], y[idx], kwargs['num_each'], kwargs['num_class'])
    del x, y, remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining), nc_score


def getNeuroCover_spilt(model, remainingData, num, **kwargs):
    (x, y) = remainingData
    interval = 100
    split = int(x.shape[0] / interval)
    ncComputor = kwargs['ncComputor']
    nc_score = np.array([])
    for i in range(0, x.shape[0], 100):
        if kwargs['method'] == 'NC':
            if i + 100 > x.shape[0]:
                new_score = ncComputor.batch_nc(x[i:])
            else:
                new_score = ncComputor.batch_nc(x[i: i+100])
        if kwargs['method'] == 'KMNC':
            if i + 100 > x.shape[0]:
                (new_score, _, _) = ncComputor.batch_kmnc(x[i:])
            else:
                (new_score, _, _) = ncComputor.batch_kmnc(x[i: i+100])
        nc_score = np.append(nc_score, new_score)
    # print(nc_score)
    assert nc_score is not None
    idx = np.argsort(nc_score)[::-1]
    select_index = idx[:num]
    x_jointrain = x[select_index]
    y_jointrain = y[select_index]
    x_remaining = np.delete(x, select_index, axis=0)
    y_remaining = np.delete(y, select_index, axis=0)
    # print(nc_score[idx[0]])
    # if np.any(nc_score==0):
    #     non_zero = np.nonzero(nc_score)[0]
    #     print(non_zero)
    #     usedIdx = idx[:non_zero.size]
    #     leftIdx = idx[non_zero.size:]
    #     np.random.shuffle(leftIdx)
    #     idx = np.concatenate([usedIdx, leftIdx])
    #
    # (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
    #     selectData(x[idx], y[idx], kwargs['num_each'], kwargs['num_class'])
    del x, y, remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining), nc_score




def getSamplesByP(model, remainingData, num, **kwargs):
    (x, y) = remainingData
    py = model.predict(x)
    pl = np.argmax(py, axis=1)
    maxp = np.squeeze(py[np.arange(len(py)), pl])
    idx = np.argsort(maxp)
    (x_jointrain, y_jointrain), (x_remaining, y_remaining) = \
        selectData(x[idx], y[idx], kwargs['num_each'], kwargs['num_class'])
    del x, y, remainingData, model
    return (x_jointrain, y_jointrain), (x_remaining, y_remaining), maxp


def getSamples(model, remainingData, num, drop_rep=50, method='var', **kwargs):
    if method == 'P':
        return getSamplesByP(model, remainingData, num, **kwargs)

    if 'var' in method:
        return getSamplesByVar(model, remainingData, num,
                               drop_rep=drop_rep,method=method, **kwargs)

    if method=='random':
        return getSamplesRandom(remainingData, num, **kwargs)

    if method=='DSA':
        return  getSamplesDSA(model, remainingData, num, **kwargs)

    if method=='LSA':
        return  getSamplesLSA(model, remainingData, num, **kwargs)
    if method== 'silhoutte':
        return getSamplesSilhoutte(model, remainingData, num, **kwargs)
    if 'KL' in method:
        return getKLDiverge(model, remainingData, num, drop_rep=drop_rep,method=method, **kwargs)

    if method=='Hist':
        return getLabelHist(model, remainingData, num, drop_rep=drop_rep, **kwargs)

    if 'NC' in method:
        return getNeuroCover(model, remainingData, num, method=method, **kwargs)


