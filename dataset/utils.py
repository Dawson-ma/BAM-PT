from sklearn.neighbors import BallTree
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm

def make_perturbed(points, rgbs, labels, gmatrix=None, radius=0.1, norms=None):
    
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
    if norms is not None:
        perturbedNorms = norms[mask, :]
    else:
        perturbedNorms = None
    if gmatrix is not None:
        gmatrix = gmatrix[mask, mask]
    
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(perturbedPoints)
    pc.colors = o3d.utility.Vector3dVector(perturbedRgbs)
    o3d.visualization.draw_geometries([pc])
    return perturbedPoints, perturbedRgbs, perturbedLabels, gmatrix, perturbedNorms

def getNorm(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)
    normals = np.asarray(pcd.normals)
    return normals

def normGen(root="shapenet", multi_index='02'):
    for cat in os.listdir(root):
        if cat[:len(multi_index)] != multi_index:
            continue
        
        print("Now processing category {}.".format(cat))
        new_folder = os.path.join(root+'_norm', cat)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
            
        cat_path = os.path.join(root, cat)
        for obj in tqdm(os.listdir(cat_path)):
            save_path = os.path.join(new_folder, obj)
            data = np.loadtxt(os.path.join(cat_path, obj))
            norms = getNorm(data[:, :3])
            
            with open(save_path, 'w') as file:
                for i in range(data.shape[0]):
                    x, y, z, r, g, b, label = data[i, :]
                    nx, ny, nz = norms[i, :]
                    file.write(f'{x} {y} {z} {r} {g} {b} {nx} {ny} {nz} {label}\n')
            file.close()
    print("Finishing normal vector generation!")                


if __name__ == "__main__":
    
    ## Perturbed Sample code
    # root="shapenetcore_partanno_segmentation_benchmark_v0_normal"
    # fileName = os.path.join(root, "03797390", '1ae1ba5dfb2a085247df6165146d5bbd.txt')
    # data = np.loadtxt(fileName)
    # N = data.shape[0]
    # points = data[:, :3]
    # rgb = data[:, 3:-1]
    # labels = data[:, -1]
    # perturbedPoints, perturbedRgbs, perturbedLabels = make_perturbed(points, rgb, labels)
    
    # Generate normal vector for shapenet
    normGen(multi_index='03')
    
    

    

