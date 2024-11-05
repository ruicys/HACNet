# **Single Image Dehazing Algorithm Driven by Contextual Collaboration and Hybrid Attention Mechanism**

---

This repository provides the implementation of our single image dehazing algorithm, which leverages contextual collaboration and a hybrid attention mechanism. 

## Framework

The architecture of **HACNet** is shown below. 
![HACNet Framework](fig/HACNet.png)

## Prepare dataset for training and evaluation

### Download Links: - [Google Drive](https://sites.google.com/view/reside-dehaze-datasets)(ITS, OTS, SOTS)
### Download Links: - [Google Drive](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/)(Dense-Haze)
### Download Links: - [Google Drive](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/)(NH-Haze)

## Data Directory Structure
### The data directory structure will be arranged as:
```plaintext
data
├── ITS
│   ├── HAZE
│   │   ├──  1_1_0.90179.png
│   │   └──  1_2_0.97842.png
│   │     
│   │            
│   └── GT
│       ├── 1.png
│       └── 2.png
│              
│           
│           
├── OTS
│   ├── HAZE
│   │   └──  0001_0.8_0.1.jpg
│   │     
│   │            
│   └── GT
│       └──0001.png
│
│  
├── SOTS
│   ├── indoor
│   │   ├── hazy
│   │   │   ├── 01_1.png
│   │   │   └── 02_1.png
│   │   └── GT
│   │       ├── 01.png
│   │       └── 02.png
│   └── outdoor
│       ├── hazy
│       │   ├── 0001_0.8_0.2.png
│       │   └── 0002_0.8_0.08.png
│       └── GT
│           ├── 0001.png
│           └── 0002.png
└── test
    ├── hazy
    │   ├──  1.png
    │   └──  2.png
    │          
    └── GT
        ├── 1_1.png
        └── 1_2.png
```

## Train
### Please download the dataset 
### For example:
```plaintext
python train.py --model 'HACNet-l' --exp 'indoor' --train_dataset 'ITS' --valid_dataset 'SOTS'
```

## Results

### Download Links: - [Google Drive](https://drive.google.com/drive/folders/1MRVD6A_CEAEH_2xqYUTtHdC20nmWbdpk?dmr=1&ec=wgc-drive-hero-goto)


## Acknowledgments

This code is based on [DehazeFormer](https://github.com/IDKiro/DehazeFormer).
