import torch
import os
import open3d as o3d
import numpy as np

color_map = {
    0: [1, 0, 0],  # Class 0: Red
    1: [0, 0, 1]  # Class 1: Green
}

def getClass(feat, dim=1):
    feat = torch.softmax(feat, dim=dim).cpu()
    feat = torch.argmax(feat, dim=dim).numpy()
    return feat

def drawPC(pts, labels, title):
    B, N, _ = pts.shape
    for i in range(B):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts[i, :, :3])
        colors = np.array([color_map[label] for label in labels[i, :]])
        pc.colors = o3d.utility.Vector3dVector(colors)
        
        pc_name = title + "_" + str(N) + "_" + str(i) + ".ply"
        save_dir = os.path.join("pc", str(N), pc_name)
        o3d.io.write_point_cloud(save_dir, pc)

pt_num = 2048
data_dir = os.path.join("output", str(pt_num))
fold_index = 0

edge_pred_dir = os.path.join(data_dir, "edge_pred_"+str(fold_index)+".pt")
pts_dir = os.path.join(data_dir, "pts_"+str(fold_index)+".pt")
seg_pred_dir = os.path.join(data_dir, "seg_pred_"+str(fold_index)+".pt")
seg_refine_dir = os.path.join(data_dir, "seg_refine_pred_"+str(fold_index)+".pt")


edge = torch.load(edge_pred_dir) # (B, 2, N)
pts = torch.load(pts_dir) # (B, N, 6)
seg_pred = torch.load(seg_pred_dir) # (B, 2, N)
seg_refine_pred = torch.load(seg_refine_dir) # (B, 2, N)


pts = pts.cpu().numpy()
edge = getClass(edge)
seg_pred = getClass(seg_pred)
seg_refine_pred = getClass(seg_refine_pred)

drawPC(pts, edge, "Edge")
drawPC(pts, seg_pred, "Segmentation")
drawPC(pts, seg_refine_pred, "Segmentation_Refine")