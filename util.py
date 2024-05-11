#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""

import torch
import torch.nn.functional as F
import numpy as np
import torchvision


def cal_loss(pred, label, smoothing=True, focal=True):
    """
    Calculate cross entropy loss, apply label smoothing if needed.
    """
    label = label.contiguous().view(-1)  # [Batch_size]
    label_one_hot = F.one_hot(label, pred.size(-1)).float().to(pred.device)  # [Batch_size, n_classes]
    pred = pred.contiguous().view(-1, pred.size(-1))  # [Batch_size, n_classes]

    if focal:
        loss = torchvision.ops.sigmoid_focal_loss(pred, label_one_hot, reduction='mean')
        return loss

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, label.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, label, reduction='mean')

    return loss


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()
