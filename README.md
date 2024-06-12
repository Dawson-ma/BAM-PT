# BAM-PT: Boundary-aware medical Point cloud Transformer for efficient segmentation of medical point clouds

This repository contains the code for the CSE 252D group project "BAM-PT: Boundary-aware medical Point cloud Transformer for efficient segmentation of medical point clouds"

## Authors
- Anand Kumar
- Tung Yen Chiang
- Tsung-Hsiang Ma
- Tung Hsiao

## Abstract

In this project, we focus on improving the efficiency of state of the art Point-based 3D intracranial aneurysm segmentation model to produce fast and precise segmentation predictions. Inspired from Point Transformer V3 which focuses on overcoming the existing trade-offs between accuracy and efficiency of processing large scale point clouds by serializing the points, we perform boundary graph based refinement after serialization. Thus, our proposed model has better processing speed while maintaining the similar performance in metrics such as IoU on the IntrA dataset. We also perform ablation studies to show the effectiveness of our proposed model.

## Dataset Preparation

Download `fileSplit`, `geo.zip` and `IntrA.zip` from [IntrA repository](https://github.com/intra3d2019/IntrA)  

Unzip `geo.zip` and `IntrA.zip` into `geo` and `IntrA` foler  

Move the unzipped `geo` folder into `IntrA/annoated/geo`  

Move the `fileSplit` into `IntrA/split`
  
Create one foler data in the code respository and add one symbolic link  

`mkdir data && ln -s Yourpath/IntrA data/IntrA`

## Installation of required packages
Step-by-step installation
```bash
# create python environment
conda create -n ept python=3.7
conda activate ept

# install dependencies
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
conda install -c anaconda h5py pyyaml -y
pip install tensorboardx

# clone this repository in your own workspace
git clone https://github.com/Dawson-Ma/BAM-PT.git
cd BAM-PT
mkdir data && ln -s Yourpath/IntrA data/IntrA

# compile cuda operations
cd point_transformer_lib
python3 setup.py build_exit install
```

## Training and Validation

### BAM-PT (Ablation) for IntrA
Download all the dependencies of ptv3 and ept.

And modify the save location in yaml file inside config/intrA.

To run use this command : `python -m PointTransformerV3.train --config config/IntrA/IntrA_pointtransformer_seg_repro.yaml sample_points 512`

### BAM-PT for IntrA

Download all the dependencies of ptv3 and ept.

And modify the save location in yaml file inside config/intrA.

To run use this command : `python -m PointTransformerV3.train_ept --config config/IntrA/IntrA_pointtransformer_seg_repro.yaml sample_points 512`

## Acknowledgements

This work is based on [point-transformer](https://github.com/POSTECH-CVLab/point-transformer) and [point-transformer-v3](https://github.com/Pointcept/PointTransformerV3) repositories. We would like to thank the authors for their work.