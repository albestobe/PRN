# DDRF

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