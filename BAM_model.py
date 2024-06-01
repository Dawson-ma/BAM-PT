from PointTransformerV3.model import PointTransformerV3
from point_transformer_lib.point_transformer_ops.point_transformer_modules import BFM_torch

class BAM_PT(nn.Module):
    def __init__(self, feat_dim, args):
        super(BAM_PT, self).__init__()
        self.seg_predictor = PointTransformerV3()
        self.edge_predictor = PointTransformerV3()
        self.refinement = BFM_torch(feat_dim, feat_dim, args.n_neighbors)
        
    def forward(self, data_dict):
        seg_pred = self.seg_predictor(data_dict)
        edge_pred = self.edge_predictor(data_dict)
        
        