import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
import numpy as np
from monai.losses import FocalLoss


class Focalloss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, logits=True, reduce=True):
        super(Focalloss, self).__init__()

    def forward(self, inputs, targets):
        loss = FocalLoss()
        #print(loss(inputs, targets))
        return loss(inputs, targets)


class CrossEntropyLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None):
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, y_input, y_target):
        #print(y_input.shape,y_target.shape)
        #print(y_input.dtype,y_target.dtype)
        # y_input = torch.log(y_input + EPSILON)
        #y_target = y_target.squeeze(1)
        return binary_cross_entropy(y_input, y_target, weight=self.weight)

class Diceloss(nn.Module):
    def __init__(self):
        super(Diceloss, self).__init__()

    def forward(self, y_input, y_target):
        #print(y_input.shape,y_target.shape)
        inter = torch.sum(y_input * y_target)
        union = torch.sum(y_input) + torch.sum(y_target)
        dice = (2. * inter + 1) / (union + 1)
        return 1 - dice


def dice(input, targets):
    inter = np.sum(input * targets)
    union = np.sum(input) + np.sum(targets)
    dice = (2. * inter + 1) / (union + 1)
    return dice

def Index(y_pred, y_ture):
    #y_pred = y_pred.detach().cpu().numpy()
    #y_ture = y_ture.detach().cpu().numpy()
    TP = np.sum(y_ture * y_pred)
    FP = np.sum(y_pred * (1 - y_ture))
    FN = np.sum((1 - y_pred) * (y_ture))
    TN = np.sum((1 - y_ture) * (1 - y_pred))
    #x,y,z =  y_ture.shape

    # print(TP,FP,FN)
    #Recall not nan
    if (TP + FN) == 0:
        Recall = 0
    else:
        Recall = TP / (TP + FN)
    #Precision not nan
    if (TP + FP) == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)
    #F_score not nan
    if (Recall + Precision) == 0:
        F_score = 0
    else:
        F_score = (2 * Recall * Precision) / (Recall + Precision)
    #ACC not nan
    if (TP + FP + FN ) ==0:
        Iou = 0
    else:
        Iou = TP / (TP + FP + FN )

    DICE = dice(y_pred, y_ture)
    ACC = (TP+TN) /(TP+TN+FP+FN)
    #print(ACC)
    #ACC = np.sum(y_pred == y_ture) / (x*y*z)

    if Recall is None:
        Recall = 0
    elif Precision is None:
        Precision = 0
    elif F_score is None:
        F_score = 0

    return DICE, ACC, Iou, F_score, Precision,Recall

def Train_dice(input, targets):
    inter = np.sum(input * targets)
    union = np.sum(input) + np.sum(targets)
    dice = (2. * inter + 1) / (union + 1)
    return dice

def Train_index(y_pred, y_ture):
    y_pred = y_pred.detach().cpu().numpy()
    y_ture = y_ture.detach().cpu().numpy()
    #x,y,z =  y_ture.shape
    TP = np.sum(y_ture * y_pred)
    FP = np.sum(y_pred * (1 - y_ture))
    FN = np.sum((1 - y_pred) * (y_ture))
    TN = np.sum((1 - y_ture) * (1 - y_pred))

    #print(TP,FP,FN)
    #Recall not nan
    if (TP + FN) == 0:
        Recall = 0
    else:
        Recall = TP / (TP + FN)
    #Precision not nan
    if (TP + FP) == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)
    #F_score not nan
    if (Recall + Precision) == 0:
        F_score = 0
    else:
        F_score = (2 * Recall * Precision) / (Recall + Precision)
    #ACC not nan
    if (TP + FP + FN ) ==0:
        Iou = 0
    else:
        Iou = TP / (TP + FP + FN )

    DICE = Train_dice(y_pred, y_ture)
    ACC = (TP+TN) /(TP+TN+FP+FN)
    #print(ACC)
    #ACC = np.sum(y_pred == y_ture) / (x*y*z)
    if Recall is None:
        Recall = 0
    elif Precision is None:
        Precision = 0
    elif F_score is None:
        F_score = 0
    return DICE, ACC, Iou, F_score, Precision, Recall


