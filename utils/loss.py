import time
import numpy as np
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix, precision_score

import torch
from torch import nn
import os
import torch.nn.functional as F


class CCCLoss(nn.Module):
    def __init__(self, reduction='mean', batch_first=True, batch_compute=False):
        super(CCCLoss, self).__init__()
        self.reduction = reduction
        self.batch_first = batch_first
        self.batch_compute = batch_compute

    def forward(self, inputs, target, mask=None):
        if mask is not None:
            if self.batch_compute:
                inputs, target = inputs[mask != 0], target[mask != 0]
            else:
                inputs, target = inputs * mask, target * mask

        if self.batch_compute:
            inputs, target = inputs.reshape(1, -1), target.reshape(1, -1)
        a_mean, b_mean = torch.mean(inputs, dim=1), torch.mean(target, dim=1)
        a_var = torch.mean(torch.square(inputs), dim=1)-torch.square(a_mean)
        b_var = torch.mean(torch.square(target), dim=1)-torch.square(b_mean)
        cor_ab = torch.mean((inputs - a_mean.unsqueeze(1))*(target - b_mean.unsqueeze(1)), dim=1)

        ccc = 2 * cor_ab / (a_var + b_var + torch.square(a_mean - b_mean) + 1e-9)
        #batch_size, length = target.shape[0], target.shape[1]
        #if not self.batch_compute:
        #    loss = torch.sum((1 - ccc) * torch.sum(mask, dim=1) / length)
        #    if self.reduction == 'mean':
        #        loss = loss / torch.sum(mask) * length
        #else:
        #    loss = torch.sum(1-ccc) if self.reduction == 'sum' else torch.mean(1-ccc)
        loss = torch.sum(1 - ccc) if self.reduction == 'sum' else torch.mean(1 - ccc)
        return loss


class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, inputs, target, mask=None):
        if mask is not None:
            return self.mse(inputs * mask, target * mask)
        else:
            return self.mse(inputs, target)


class CELoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=-1):
        super(CELoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, target, mask=None):
        #logits: [B, 22, L], target: [B, L], mask: [B, L]
        if mask is not None:
            target[mask == 0] = -1
        return self.ce(inputs, target)


class MultipleLoss(nn.Module):
    def __init__(self, reduction='mean', loss_names=('mse', ), weights=(1, )):
        super(MultipleLoss, self).__init__()
        self.reduction = reduction
        self.loss_names = loss_names
        self.weights = weights
        self.loss_functions = []
        for name in loss_names:
            if 'mse' in name:
                self.loss_functions.append(MSELoss(reduction='sum'))
            elif 'ccc' in name:
                batch_compute = True if 'batch' in name else False
                self.loss_functions.append(CCCLoss(reduction='mean', batch_compute=batch_compute))
            elif 'ce' in name:
                self.loss_functions.append(CELoss(reduction='mean', ignore_index=-1))

    def forward(self, inputs, target, mask=None, logits=None, cls_target=None):
        loss = 0
        ce_loss = 0
        for i, (name, function) in enumerate(zip(self.loss_names, self.loss_functions)):
            if name in ['vmse', 'vccc', 'batch_vccc']:
                loss = self.weights[i] * function(inputs[..., 0], target[..., 0], mask[..., 0]) + loss
            elif name in ['amse', 'accc', 'batch_accc']:
                loss = self.weights[i] * function(inputs[..., -1], target[..., -1], mask[..., -1]) + loss
            elif name == 'ce':
                for j in range(inputs.shape[-1]):
                    loss = self.weights[i] * \
                           function(logits[..., j], cls_target[..., j], mask[..., j]) / inputs.shape[-1] + loss
        #            ce_loss = self.weights[i] * \
        #                   function(logits[..., j], cls_target[..., j], mask[..., j]) / inputs.shape[-1] + ce_loss
            else:
                for j in range(inputs.shape[-1]):
                    loss = self.weights[i] * \
                           function(inputs[..., j], target[..., j], mask[..., j]) / inputs.shape[-1] + loss

        #print(loss, ce_loss)
        
        loss = (torch.tensor(0.0, requires_grad=True) if loss==0 else loss)##不加这句会报错
        
        return loss
