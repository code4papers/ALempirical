# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
import argparse
# import model
import math
from scipy import special
import numpy as np

import sys


# Input:
# model: the torch model
# input: the input at current stage
#        Torch tensor with size (Batchsize,length)
# Output: score with size (batchsize, length)
# def random(model, inputs, pred, classes):
#     losses = torch.rand(inputs.size()[0], inputs.size()[1])
#     return losses
#     # Output a random list
#
#
def replaceone(model, inputs, pred, classes):
    losses = np.zeros((inputs.shape[0], inputs.shape[1]))
    for i in range(inputs.shape[1]):
        tempinputs = inputs.copy()
        # print(tempinputs)
        tempinputs[:, i] = 2
        # print(tempinputs)
        tempoutput = model.predict(tempinputs)
        # print(tempoutput)
        # print(pred)
        left_one = -special.xlogy(tempoutput[0], pred[0])
        raight_onbe = special.xlogy(1-tempoutput[0], 1-pred[0])
        # print(left_one - raight_onbe)
        # print(raight_onbe)
        losses[:, i] = np.sum(-special.xlogy(tempoutput[0], pred[0]) - special.xlogy(1-tempoutput[0], 1-pred[0]))
    return losses
#
#
# def temporal(model, inputs, pred, classes):
#     losses1 = torch.zeros(inputs.size()[0], inputs.size()[1])
#     dloss = torch.zeros(inputs.size()[0], inputs.size()[1])
#     for i in range(inputs.size()[1]):
#         tempinputs = inputs[:, :i + 1]
#         with torch.no_grad():
#             tempoutput = torch.exp(model(tempinputs))
#         losses1[:, i] = tempoutput.gather(1, pred.view(-1, 1)).view(-1)
#     dloss[:, 0] = losses1[:, 0] - 1.0 / classes
#     for i in range(1, inputs.size()[1]):
#         dloss[:, i] = losses1[:, i] - losses1[:, i - 1]
#     return dloss


def temporaltail(model, inputs, pred, classes):
    # losses1 = torch.zeros(inputs.size()[0], inputs.size()[1])
    # dloss = torch.zeros(inputs.size()[0], inputs.size()[1])
    # for i in range(inputs.size()[1]):
    #     tempinputs = inputs[:, i:]
    #     with torch.no_grad():
    #         tempoutput = torch.exp(model(tempinputs))
    #     losses1[:, i] = tempoutput.gather(1, pred.view(-1, 1)).view(-1)
    # dloss[:, -1] = losses1[:, -1] - 1.0 / classes
    # for i in range(inputs.size()[1] - 1):
    #     dloss[:, i] = losses1[:, i] - losses1[:, i + 1]

    losses1 = np.zeros((inputs.shape[0], inputs.shape[1]))
    dloss = np.zeros((inputs.shape[0], inputs.shape[1]))
    print(inputs)
    for i in range(inputs.shape[1]):
        tempinputs = inputs[:, i:]
        print(tempinputs)
        print(model.predict(tempinputs))
        tempoutput = np.exp(model.predict(tempinputs))
        print(tempoutput)
        losses1[:, i] = tempoutput.gather(1, pred.view(-1, 1)).view(-1)
    dloss[:, -1] = losses1[:, -1] - 1.0 / classes
    for i in range(inputs.shape[1] - 1):
        dloss[:, i] = losses1[:, i] - losses1[:, i + 1]
    return dloss


# def combined(model, inputs, pred, classes):
#     temp = temporal(model, inputs, pred, classes)
#     temptail = temporaltail(model, inputs, pred, classes)
#     return (temp + temptail) / 2
#
#
# def grad(model, inputs, pred, classes):
#     losses1 = torch.zeros(inputs.size()[0], inputs.size()[1])
#     dloss = torch.zeros(inputs.size()[0], inputs.size()[1])
#     if isinstance(model, torch.nn.DataParallel):
#         model = model.module
#     model.train()
#     embd, output = model(inputs, returnembd=True)
#     # embd.retain_grad()
#     loss = F.nll_loss(output, pred)
#
#     loss.backward()
#     score = (inputs <= 2).float()
#     score = -score
#     score = embd.grad.norm(2, dim=2) + score * 1e9
#     return score
#
#
# def grad_unconstrained(model, inputs, pred, classes):
#     losses1 = torch.zeros(inputs.size()[0], inputs.size()[1])
#     dloss = torch.zeros(inputs.size()[0], inputs.size()[1])
#     if isinstance(model, torch.nn.DataParallel):
#         model = model.module
#     model.train()
#     embd, output = model(inputs, returnembd=True)
#     loss = F.nll_loss(output, pred)
#
#     loss.backward()
#     score = embd.grad.norm(2, dim=2)
#     return score


def scorefunc(name):
    # if "temporal" in name:
    #     return temporal
    if "tail" in name:
        return temporaltail
    # elif "combined" in name:
    #     return combined
    elif "replaceone" in name:
        return replaceone
    # elif "random" in name:
    #     return random
    # elif 'ucgrad' in name:
    #     return grad_unconstrained
    # elif "grad" in name:
    #     return grad
    # else:
    #     print('No scoring function found')
    #     sys.exit(1)
