# _Install package for PointTransformerV3_ #

## _install basic package_ ##
```bash
pip install ninja h5py pyyaml sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm
```

## _install pytorch sub-package_ ##
```bash
pip install torch_cluster-1.6.1+pt20cu118-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.1+pt20cu118-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.17+pt20cu118-cp39-cp39-linux_x86_64.whl
pip install torch-geometric
```

## _install pointops_ ##
```bash
cd /path/to/PointTransformerV3/
cd libs/pointops
python setup.py install
cd ../../
pip install spconv-cu118
```

## _install open3d for visualization_ ##
```bash
pip install open3d
```

## _install flash attention_ ##
```bash
pip install packaging wheel
apt-get update && apt-get install -y git
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

# _install for EPT-Net_ #

## _install basic package_ ##
```bash
pip install tensorboardx wandb trimesh
cd point_transformer_lib
python setup.py build_ext install
```

# _Run the training code_ #
```bash
python -m BAM_train --config config/ShapeNet/ShapeNet_BAM_PT.yaml sample_points 512
```

# PTV3 for IntrA
Download all the dependencies of ptv3 and ept.

And modify the save location in yaml file inside config/intrA.

To run use this command : `python -m PointTransformerV3.train --config config/IntrA/IntrA_pointtransformer_seg_repro.yaml sample_points 512`