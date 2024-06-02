# _Install package for PointTransformerV3_ #

## _install basic package_ ##
```bash
pip install ninja h5py pyyaml sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm open3d
```

## _install pytorch sub-package_ ##
The file downloading website is here:
`https://download.pytorch.org/whl/torch_stable.html`
```bash
pip install torch_cluster-1.6.1+pt20cu118-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.1+pt20cu118-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.17+pt20cu118-cp39-cp39-linux_x86_64.whl
```

## _install pointops_ ##
```bash
cd /path/to/PointTransformerV3/
cd libs/pointops
python setup.py install
cd ../../
pip install spconv-cu118 torch-geometric
```

## _install flash attention_ ##
```bash
pip install packaging wheel
apt-get update && apt-get install -y git
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

# _Install packages for EPT-Net_ #

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
Remember to run the code in the root folder, and the sample_points is for training. When running the test code, please change to test_points.
## _If encountering problem OSError: libGL.so.1: cannot open shared object file: No such file or directory_ ##
```bash
ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
DEBIAN_FRONTEND=noninteractive apt-get install gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils wget -y
```

# PTV3 for IntrA
Download all the dependencies of ptv3 and ept.

And modify the save location in yaml file inside config/intrA.

To run use this command : `python -m PointTransformerV3.train --config config/IntrA/IntrA_pointtransformer_seg_repro.yaml sample_points 512`