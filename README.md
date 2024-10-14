# PRN: Progressive Reasoning Network and Its Image Completion Applications

## Paper
Ancient murals hold significant historical, cultural, scientific, and artistic values, but many suffer from issues such as pigment shedding or missing parts. As a potential solution, deep learning-based inpainting has achieved impressive results on natural images, but it has proven unsatisfactory in repairing images of damaged murals due to data shift and low modeling efficiency. In this paper, we propose a novel progressive reasoning network for image completion of ancient murals, taking into account the mural painting process. The proposed network comprises a luminance reasoning module, a sketch reasoning module, and a color fusion module. Both the first two modules are built on the double-codec framework to infer the luminance and sketch information of missing areas, respectively. The final module introduces a paired-associate learning framework for recovering color images. This network first uses two parallel complementary paths to estimate the respective luminance and sketch maps of a damaged image, and then combines them to synthesize the complete color image. Experimental results demonstrate that the proposed network recovers clearer structures and more realistic colors, generally outperforming current state-of-the-art methods in repairing damaged images, both quantitatively and qualitatively.   

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
year={2024}
}
```
