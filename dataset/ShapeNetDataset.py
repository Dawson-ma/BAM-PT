import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
import json
from dataset.utils import make_perturbed, getNorm
from point_transformer_lib.point_transformer_ops.point_transformer_utils import FPS

# Download link for ShapeNet: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ShapeNetDataset(Dataset):
    def __init__(self, root="shapenet", mode='train', transform=None, pcSize=None,
                 uniform=False, use_norms=False, K=4, perturbed=False, radius=0.2):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.pcSize = pcSize
        self.uniform = uniform
        self.use_norms = use_norms
        self.K = K
        self.perturbed = perturbed
        self.radius = radius
        
        jsonFile = os.path.join(root, "train_test_split", "shuffled_"+mode+"_file_list.json")
        with open(jsonFile, "r") as file:
            data = json.load(file)
            
        self.data = [d.split("/")[1:] for d in data]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        # Data loading
        fileName = os.path.join(self.root, self.data[index][0], self.data[index][1]+'.txt')
        data = np.loadtxt(fileName)
        
        points = data[:, :3]
        rgbs = data[:, 3:-1]
        # norms = getNorm(points)
        labels = data[:, -1]
        
        if self.perturbed:
            points, rgbs, labels, norms = make_perturbed(points, rgbs, labels, radius=self.radius, norms=None)
        
        N = points.shape[0]
        point_idxs = range(N)
        gmatrix = torch.cdist(torch.from_numpy(points), torch.from_numpy(points))
        
        # Sampling
        if N >= self.pcSize:
            if self.uniform:
                points_cuda = torch.from_numpy(points).float().unsqueeze(0)
                selected_points_idxs = FPS(points_cuda, self.pcSize).squeeze().numpy().astype(np.int64)
        else:
            if self.uniform:
                points_cuda = torch.from_numpy(points).float().unsqueeze(0)
                scale = self.pcSize // N
                extra = self.pcSize % N
                extra_idxs = FPS(points_cuda, extra).squeeze().numpy().astype(np.int64)
                selected_points_idxs = np.concatenate((np.array(list(point_idxs)*scale).astype(np.int64), extra_idxs))
            
            
        selected_points = points[selected_points_idxs]
        selected_rgbs = rgbs[selected_points_idxs]
        # selected_norms = norms[selected_points_idxs]
        selected_gmatrix = gmatrix[selected_points_idxs, :][:, selected_points_idxs]
        selected_labels = labels[selected_points_idxs]
            
        selected_points = pc_normalize(selected_points)
        if self.use_norms:
            selected_points = np.concatenate((selected_points, selected_rgbs), axis=1)
        if self.transform is not None:
            selected_points = self.transform(selected_points).float()
        else:
            selected_points = torch.from_numpy(selected_points).float()
                
        selected_labels = torch.from_numpy(selected_labels).long()
        selected_gmatrix = selected_gmatrix.float()
        selected_edge_labels, edgeweights = self.get_edge_label(selected_points_idxs, selected_labels, selected_gmatrix, self.K)
        return selected_points, selected_labels, selected_edge_labels, edgeweights, selected_gmatrix, selected_points_idxs
    
    def get_edge_label(self, idxs, labels, gmatrix, k): 
        _, indices, reverse_indices = np.unique(idxs, return_index=True, return_inverse=True)
        unique_labels = labels[indices]
        unique_gmatrix = gmatrix[indices, :][:, indices]
        edge_labels = torch.zeros(unique_labels.shape[0])
        idxs_neighbor = unique_gmatrix.argsort(dim=-1)[:, :k] # (N, K)
        gts_neighbor = torch.gather(unique_labels[None, :].repeat(idxs_neighbor.shape[0], 1), 1, idxs_neighbor) # (N, K)
        gts_neighbor_sum = gts_neighbor.sum(dim=-1)
        edge_mask = torch.logical_and(gts_neighbor_sum!=0, gts_neighbor_sum!=k)
        edge_labels[edge_mask] = 1
        edge_labels = edge_labels[reverse_indices]
        edgeweights = torch.histc(edge_labels, bins=2, min=0, max=1)
        edgeweights = edgeweights / torch.sum(edgeweights)
        edgeweights = (edgeweights.max() / edgeweights) ** (1/3)
        edge_labels = edge_labels.long()
        return edge_labels, edgeweights

    
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ShapeNetDataset(transform=transform, pcSize=1024)