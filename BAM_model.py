from PointTransformerV3.model import PointTransformerV3
from point_transformer_lib.point_transformer_ops.point_transformer_modules import BFM_torch
from torch import nn as nn
from utils.timer import Timer
import torch.nn.functional as F

class BAM_PT(nn.Module):
    def __init__(self, args):
        super(BAM_PT, self).__init__()
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
        
    def forward(self, data_dict, gmatrix, idxs):
        B = data_dict['batch'][-1]+1
        seg_feat = self.seg_predictor(data_dict)['feat'].view(B, -1, self.feat_dim).transpose(1, 2).contiguous()
        edge_feat = self.edge_predictor(data_dict)['feat'].view(B, -1, self.feat_dim).transpose(1, 2).contiguous()
        
        seg_pred = self.seg_fc_layer(seg_feat)
        edge_pred = self.edge_fc_layer(edge_feat)
        
        seg_refine_features = self.refinement(seg_feat, edge_pred, gmatrix, idxs)
        seg_refine_preds = self.seg_refine_fc_layer(seg_refine_features.transpose(1, 2).contiguous())
        seg_embed = F.normalize(self.proj_layer(seg_feat), p=2, dim=1)
        return seg_pred, seg_refine_preds, seg_embed, edge_pred
        
        
        
        