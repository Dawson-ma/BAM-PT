from PointTransformerV3.model import PointTransformerV3
from point_transformer_lib.point_transformer_ops.point_transformer_modules import BFM_torch
from torch import nn as nn
from utils.timer import Timer
import torch.nn.functional as F
import torch
import numpy as np
from PointTransformerV3.model import PointTransformerV3, PointSequential, Block, SerializedUnpooling, Point

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
    
class BAM_PT(PointTransformerV3):
    def __init__(self,
    in_channels=6,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024, 1024, 1024, 1024),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    pre_norm=True,
    shuffle_orders=True,
    enable_rpe=False,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    cls_mode=False,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"), args=None):
        super().__init__(in_channels, order, stride, enc_depths, enc_channels, enc_num_head, enc_patch_size, dec_depths, dec_channels, dec_num_head, dec_patch_size, mlp_ratio, qkv_bias, qk_scale, attn_drop, proj_drop, drop_path, pre_norm, shuffle_orders, enable_rpe, enable_flash, upcast_attention, upcast_softmax, cls_mode, pdnorm_bn, pdnorm_ln, pdnorm_decouple, pdnorm_adaptive, pdnorm_affine, pdnorm_conditions)
        self.pointCloudSTN = STN3d()
        self.refinement = BFM_torch(dec_channels[0], dec_channels[0], args.n_neighbors)
        act_layer = nn.GELU
        bn_layer = nn.BatchNorm1d
        ln_layer = nn.LayerNorm
        
        self.seg_fc = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0], bias=False),
            bn_layer(dec_channels[0]),
            act_layer(),
            nn.Linear(dec_channels[0], 2),
        )
        
        self.edge_fc = nn.Sequential(
                nn.Linear(dec_channels[0], dec_channels[0], bias=False),
                bn_layer(dec_channels[0]),
                act_layer(),
                nn.Linear(dec_channels[0], 2),
            )
        
        self.proj_layer = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0], bias=False),
            bn_layer(dec_channels[0]),
            act_layer(),
            nn.Linear(dec_channels[0], 2),
        ) 
        
        self.seg_refine_fc = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0], bias=False),
            bn_layer(args.sample_points),
            act_layer(),
            nn.Linear(dec_channels[0], 2),
        )
        
    def forward(self, pts, gts, gmatrix, idxs):
        # x, normals = pts[:, :, :3], pts[:, :, 3:]
        # x = x.transpose(2, 1)
        # trans = self.pointCloudSTN(x)
        # x = x.transpose(2, 1)
        # x = torch.bmm(x, trans)
        # pts = torch.cat((x, normals), dim=-1)
        
        # Create dict
        batches = np.arange(pts.shape[0])
        batches = np.tile(batches, (pts.shape[1], 1)).T.flatten()
        batches = torch.tensor(batches).long()
        
        data_dict = {'batch': batches.cuda(), 'feat': pts.flatten(end_dim=1).cuda().to(torch.float32), 'coord': pts.flatten(end_dim=1)[:,0:3].cuda().to(torch.float32), 'labels': gts.flatten().cuda(), 'grid_size': torch.tensor(0.01).to(torch.float32)}
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        point_seg = self.dec(point)
        
        B, N, _ = pts.shape
        seg_pred = self.seg_fc(point_seg.feat) # (4096, 2)
        edge_pred = self.edge_fc(point_seg.feat)# (4096, 2)
        
        # (8, 512, 64)
        seg_refine_features = self.refinement(point_seg.feat.view(B, N, -1).transpose(1, 2), edge_pred.view(B, N, -1).transpose(1, 2).contiguous(), gmatrix, idxs)
        # (8, 2, 512)
        seg_refine_preds = self.seg_refine_fc(seg_refine_features.contiguous()).transpose(1, 2)
        # (8, 2, 512)
        seg_embed = F.normalize(self.proj_layer(point_seg.feat), p=2, dim=1).reshape(B, N, -1).transpose(1, 2)
        return seg_pred.view(B, N, -1).transpose(1, 2), seg_refine_preds, seg_embed, edge_pred.view(B, N, -1).transpose(1, 2)