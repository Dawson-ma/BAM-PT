import os
import time
import random
import numpy as np
import logging
import argparse
import shutil

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataset import IntrADataset
from dataset import data_utils as d_utils

from utils import config
from utils.tools import cal_IoU_Acc_batch, get_contra_loss, record_statistics
from PointTransformerV3.model import PointTransformerV3
from BAM_model import BAM_PT
from point_transformer_lib.point_transformer_ops.point_transformer_modules import BFM_torch

test_fold = [0, 1, 2, 3, 4]
sample_points = 512
num_edge_neighbor = 4
refinement = BFM_torch(2, 2, num_edge_neighbor)
model = PointTransformerV3().cuda()

train_transforms = transforms.Compose(
    [d_utils.PointcloudToTensor(),
    d_utils.PointcloudScale(),
    d_utils.PointcloudRotate(),
    d_utils.PointcloudRotatePerturbation(),
    d_utils.PointcloudTranslate(),
    d_utils.PointcloudJitter(),
    d_utils.PointcloudRandomInputDropout(),]
)
path = "dataset/IntrA/"
train_data = IntrADataset.IntrADataset_PTv3(path, sample_points, True, True, 
                    test_fold=test_fold, num_edge_neighbor=num_edge_neighbor, mode='train', transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=8,
                            shuffle=True, num_workers=8, pin_memory=True, drop_last=True)


for batch_i, (norms, coords, labels, g_mat, idxs) in enumerate(train_loader):
    batches = np.arange(norms.shape[0])
    batches = np.tile(batches, (norms.shape[1], 1)).T.flatten()
    batches = torch.tensor(batches).long()

    data_dict = {'batch': batches.cuda(), 'feat': coords.flatten(end_dim=1).cuda().to(torch.float32), 'coord': coords.flatten(end_dim=1)[:,0:3].cuda().to(torch.float32), 'labels': labels.flatten().cuda(), 'grid_size': torch.tensor(0.001).to(torch.float32)}
    results = model(data_dict)
    seg_refine_features = refinement(results['feat'].view(8, -1, 2), edge_preds, g_mat, idxs)
    break