from sklearn.neighbors import BallTree
import numpy as np
import open3d as o3d
import os

def make_perturbed(points, rgbs, labels, gmatrix=None, radius=0.1):
    
    pc_ori = o3d.geometry.PointCloud()
    pc_ori.points = o3d.utility.Vector3dVector(points)
    pc_ori.colors = o3d.utility.Vector3dVector(rgbs)
    o3d.visualization.draw_geometries([pc_ori])

    treeModel = BallTree(points)
    centerIndex = np.random.randint(len(points))
    center = points[centerIndex]
    selectIndex = treeModel.query_radius(center.reshape(1, -1), r=radius)[0]
    
    mask = np.ones(points.shape[0], dtype=bool)
    mask[selectIndex] = False
    perturbedPoints = points[mask, :]
    perturbedRgbs = rgbs[mask, :]
    perturbedLabels = labels[mask]
    if gmatrix is not None:
        gmatrix = gmatrix[mask, mask]
    
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(perturbedPoints)
    pc.colors = o3d.utility.Vector3dVector(perturbedRgbs)
    o3d.visualization.draw_geometries([pc])
    return perturbedPoints, perturbedRgbs, perturbedLabels, gmatrix

if __name__ == "__main__":
    root="shapenetcore_partanno_segmentation_benchmark_v0_normal"
    fileName = os.path.join(root, "03797390", '1ae1ba5dfb2a085247df6165146d5bbd.txt')
    data = np.loadtxt(fileName)
    N = data.shape[0]
    points = data[:, :3]
    rgb = data[:, 3:-1]
    labels = data[:, -1]
    perturbedPoints, perturbedRgbs, perturbedLabels = make_perturbed(points, rgb, labels)
    
    

    

