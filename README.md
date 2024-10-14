# PRN: Progressive Reasoning Network and Its Image Completion Applications

## Abstract
Ancient murals embody profound historical, cultural, scientifc, and artistic values, yet many are aficted with challenges such as pigment shedding or missing parts. While deep learning-based completion techniques have yielded remarkable results in restoring natural images, their application to damaged murals has been unsatisfactory due to data shifts and limited modeling efcacy. This paper proposes a novel progressive reasoning network designed specifcally for mural image completion, inspired by the mural painting process. The proposed network comprises three key modules: a luminance reasoning module, a sketch reasoning module, and a color fusion module. The frst two modules are based on the double-codec framework, designed to infer missing areas’ luminance and sketch information. The fnal module then utilizes a paired-associate learning approach to reconstruct the color image. This network utilizes two parallel, complementary pathways to estimate the luminance and sketch maps of a damaged mural. Subsequently, these two maps are combined to synthesize a complete color image. Experimental results indicate that the proposed network excels in restoring clearer structures and more vivid colors, surpassing current state-of-the art methods in both quantitative and qualitative assessments for repairing damaged images. 

Keywords Image completion, Image inpainting, Deep learning, Ancient murals, Pigment shedding

[[`article`](https://doi.org/10.1038/s41598-024-72368-1)]

![PRN](/Figures/Network)  
Network architecture of our PRN.


## Requirements
Python==3.9

Pytorch==1.11.0+cu113


## Image inpainting using existing models
To complete mural Image Inpainting, place the images to be repaired into the folder **./testimg**, place the mask into the folder **./mask**, (we recommend naming both "1.jpg") use
```
python test.py
```   
The final inpainting result can be viewed in the folder **./finalresult**.

## Use your own data set for network training
The training of the Brightness Map model is the same as that of the Sketch Map model, use
```
cd brightness (cd sketch)
python run.py --data_root [your image path]  --mask_root [your mask path] --mask_mode 0 --model_path [Pre-trained model path] 
```

To train the color fusion module
```
cd PIAFusion_pytorch-masterGPU3 
python train_fusion_model.py --dataset_path 'datasets/msrs_train' --epochs 20 --pretrained_module 'pretrained/fusion_model_epoch_9.pth' --batch_size 6 --lr 0.0001  --save_path 'pretrained' 
```
The directory in the **msrs_train** folder must have the following format:
```
└─msrs_train
    ├─brightresult
    ├─cartoonresult
    ├─originalimage
    └─originalmask
```
where **brightresult** contains the outputs of brightness reasoning module, **cartoonresult** contains the outputs of sketch reasoning module, **originalimage** contains the original images, and **originalmask** contains the corresponding masks.

## Citation         
```
@article{ZhangYQ24,
title = {PRN: Progressive Reasoning Network and Its Image Completion Applications},
author = {Yongqin Zhang, Xiaoyu Wang, Panpan Zhu, Xuan Lu, Jinsheng Xiao, Wei Zhou, Zhan Li, and Xianlin Peng},
journal = {Scientific Reports},    
volume = {14},
pages = {23519},
year={2024}
}
```
