# Edge-oriented Point-cloud Transformer for 3D Intracranial Aneurysm Segmentation
by [Yifan Liu](https://github.com/yifliu3)


## 1.Introduction
This repository is for our MICCAI 2022 paper "Edge-oriented Point cloud Transformer for 3D Intracranial Aneurysm Segmentation"  

## 2.Data Preparation
Download `fileSplit`, `geo.zip` and `IntrA.zip` from [IntrA repository](https://github.com/intra3d2019/IntrA)  

Unzip `geo.zip` and `IntrA.zip` into `geo` and `IntrA` foler  

Move the unzipped `geo` folder into `IntrA/annoated/geo`  

Move the `fileSplit` into `IntrA/split`
  
Create one foler data in the code respository and add one symbolic link  

`mkdir data && ln -s Yourpath/IntrA data/IntrA`

## 3. Installation
### Requirements
- python 3.7
- pytorch 1.7
- h5py
- pyyaml
- tensorboardx

### Step-by-step installation
```bash
# create python environment
conda create -n ept python=3.7
conda activate ept

# install dependencies
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
conda install -c anaconda h5py pyyaml -y
pip install tensorboardx

# clone this repository in your own workspace
git clone https://github.com/CityU-AIM-Group/EPT.git
cd EPT
mkdir data && ln -s Yourpath/IntrA data/IntrA

# compile cuda operations
cd point_transformer_lib
python3 setup.py build_exit install

```

## 4. Train/test the Model 
To separately train and test you can use the commands below (take 512 sampling as an example):  
Train:   
`python -m tool.train --config config/IntrA/IntrA_pointtransformer_seg_repro sample_points 512`  
Test:  
`python -m tool.test --config config/IntrA/IntrA_pointtransformer_seg_repro sample_points 512`  


Or you can use the bash scipt to run train.py and test.py sequentially:  
`sh tool/ept.sh IntrA pointtransformer_seg_repro`  

The trained models are provided in [Google Drive](https://drive.google.com/drive/folders/1wThn1dBmQk36-suSJOq5T8UJq3GPQ6QF?usp=sharing)

## 5. Citation
If you find this work useful for your research, please cite our paper:
```
@inproceedings{liu2022,
  title={Edge-oriented Point-cloud Transformer for 3D Intracranial Aneurysm Segmentation},
  author={Yifan Liu, Jie Liu and Yixuan Yuan},
  booktitle= {MICCAI},
  year={2022}
}
```

## Results EPT

### 512:

Norm:

        Best mIoU is 0.9313, vIoU is 0.9748, aIoU is 0.8879
[2024-06-01 04:00:24,833 INFO train.py line 69 5643] => Final mIoU is 0.9294, vIoU is 0.9672, aIoU is 0.8915

Ablate:

        Best mIoU is 0.9234, vIoU is 0.9713, aIoU is 0.8755
[2024-06-01 04:24:44,633 INFO train.py line 69 5645] => Final mIoU is 0.9204, vIoU is 0.9632, aIoU is 0.8775

### 1024:

Norm:

        Best mIoU is 0.9220, vIoU is 0.9661, aIoU is 0.8780
[2024-06-01 02:49:48,651 INFO train.py line 69 5637] => Final mIoU is 0.9289, vIoU is 0.9673, aIoU is 0.8906

Ablate:

        Best mIoU is 0.9234, vIoU is 0.9698, aIoU is 0.8769
[2024-06-02 21:21:03,105 INFO train.py line 69 5636] => Final mIoU is 0.9214, vIoU is 0.9642, aIoU is 0.8786


### 2048:

Norm:

        Best mIoU is 0.9186, vIoU is 0.9638, aIoU is 0.8734
[2024-06-01 08:34:51,107 INFO train.py line 69 5645] => Final mIoU is 0.9182, vIoU is 0.9616, aIoU is 0.8748

Ablate:

        Best mIoU is 0.9196, vIoU is 0.9647, aIoU is 0.8744
[2024-06-01 08:17:35,104 INFO train.py line 69 5637] => Final mIoU is 0.9157, vIoU is 0.9611, aIoU is 0.8704


## Results PTV3

### 512:

With EPT:

                Best mIoU is 0.9318, vIoU is 0.9689, aIoU is 0.8947

Without EPT:

                Best mIoU is 0.9121, vIoU is 0.9576, aIoU is 0.8665


## Inference Time

### EPT

Normal: 

| Points    | Time/batch |
| -------- | :-------: |
| 512  | 0.0103    |
| 1024 | 0.0124    |
| 2048 | 0.0219   |

Ablate: 

| Points    | Time/batch |
| -------- | :-------: |
| 512  | 0.0049   |
| 1024 | 0.0092    |
| 2048 | 0.0104   |


### PTV3

With EPT: 

| Points    | Time/batch |
| -------- | :-------: |
| 512  | 0.0108 |
| 1024 | 0.0112  |
| 2048 | 0.0138  |

WIthout EPT: 

| Points    | Time/batch |
| -------- | :-------: |
| 512  | 0.0071 |
| 1024 | 0.0074  |
| 2048 | 0.0084 |



## 6. Acknowledgement
This work is based on [point-transformer](https://github.com/POSTECH-CVLab/point-transformer).

