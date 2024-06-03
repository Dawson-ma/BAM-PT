from PointTransformerV3.model import PointTransformerV3
from point_transformer_lib.point_transformer_ops.point_transformer_modules import BFM_torch
from torch import nn as nn
from utils.timer import Timer
import torch.nn.functional as F
import torch
import numpy as np

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1).float()
        self.conv2 = nn.Conv1d(64, 128, 1).float()
        self.conv3 = nn.Conv1d(128, 1024, 1).float()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bias = nn.Parameter(torch.zeros(1, 3, 3))
        
    def forward(self, x):
        batchNumber = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        x = x.view(-1, 3, 3)
        bias = self.bias.repeat(batchNumber, 1, 1)
        x += bias
        return x
    
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1).float()
        self.conv2 = nn.Conv1d(64, 128, 1).float()
        self.conv3 = nn.Conv1d(128, 1024, 1).float()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bias = nn.Parameter(torch.zeros(1, k, k))
        self.k = k
        
    def forward(self, x):
        batchNumber = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        x = x.view(-1, self.k, self.k)
        bias = self.bias.repeat(batchNumber, 1, 1)
        x += bias
        return x

class BAM_PT(nn.Module):
    def __init__(self, args):
        super(BAM_PT, self).__init__()
        self.pointCloudSTN = STN3d()
        self.featureSTN = STNkd(k=6)
        self.seg_predictor = PointTransformerV3()
        self.edge_predictor = PointTransformerV3()
        self.feat_dim = args.feat_dim
        self.cls_num = args.cls_num
        self.refinement = BFM_torch(self.feat_dim, self.feat_dim, args.n_neighbors)
        
        self.seg_fc_layer = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(self.feat_dim, self.cls_num, kernel_size=1),
        )

        self.edge_fc_layer = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(self.feat_dim, 2, kernel_size=1),
        )
        
        self.proj_layer = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=1),
        )
        
        self.seg_refine_fc_layer = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(self.feat_dim, self.cls_num, kernel_size=1),
        )
        
    def forward(self, pts, gts, gmatrix, idxs):
        
        x, normals = pts[:, :, :3], pts[:, :, 3:]
        
        x = x.transpose(2, 1)
        trans1 = self.pointCloudSTN(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans1)
        
        # Create dict
        batches = np.arange(pts.shape[0])
        batches = np.tile(batches, (pts.shape[1], 1)).T.flatten()
        batches = torch.tensor(batches).long()
        pts = torch.cat((x, normals), dim=-1)

        data_dict = {'batch': batches.cuda(), 'feat': pts.flatten(end_dim=1).cuda().to(torch.float32), 'coord': pts.flatten(end_dim=1)[:,0:3].cuda().to(torch.float32), 'labels': gts.flatten().cuda(), 'grid_size': torch.tensor(0.01).to(torch.float32)}
        
        
        B = pts.shape[0]
        seg_feat = self.seg_predictor(data_dict)['feat'].view(B, -1, self.feat_dim).transpose(1, 2).contiguous()
        edge_feat = self.edge_predictor(data_dict)['feat'].view(B, -1, self.feat_dim).transpose(1, 2).contiguous()
        
        seg_pred = self.seg_fc_layer(seg_feat)
        edge_pred = self.edge_fc_layer(edge_feat)
        
        seg_refine_features = self.refinement(seg_feat, edge_pred, gmatrix, idxs)
        seg_refine_preds = self.seg_refine_fc_layer(seg_refine_features.transpose(1, 2).contiguous())
        seg_embed = F.normalize(self.proj_layer(seg_feat), p=2, dim=1)
        return seg_pred, seg_refine_preds, seg_embed, edge_pred