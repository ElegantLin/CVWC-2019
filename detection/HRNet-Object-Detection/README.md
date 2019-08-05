# High-resolution networks (HRNets) for object detection
## Introduction
This is the official code of [High-Resolution Representations for Object Detection](https://arxiv.org/pdf/1904.04514.pdf). We extend the high-resolution representation (HRNet) [1] by augmenting the high-resolution representation by aggregating the (upsampled) representations from all the parallel
convolutions, leading to stronger representations. We build a multi-level representation from the high resolution and apply it to the Faster R-CNN, Mask R-CNN and Cascade R-CNN framework. This proposed approach achieves superior results to existing single-model networks 
on COCO object detection. The code is based on [mmdetection](https://github.com/open-mmlab/mmdetection)

<div align=center>

![](images/hrnetv2p.png)

</div>

## News

We've involved **SyncBatchNorm** and **Multi-scale training(We provided two kinds of implementation)** in HRNetV2 now! After trained with multiple scales and SyncBN, the detection models
 obtain better performance. Code and models have been updated already! 

## Performance
### ImageNet pretrained models
HRNetV2 ImageNet pretrained models are now available! Codes and pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification)

All models are trained on COCO *train2017* set and evaluated on COCO *val2017* set. Detailed settings or configurations are in [`configs/hrnet`](configs/hrnet).

**Note:** Models are trained with the newly released code and the results have minor differences with that in the paper. 
Current results will be updated soon and more models and results are comming.

### Faster R-CNN
|Backbone|#Params|GFLOPs|lr sched|SyncBN|MS train|mAP|pretrained model|detection model|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| HRNetV2-W18 |26.2M|159.1| 1x |N|N| 36.1 | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [FasterR-CNN-HR18-1x.pth](https://1drv.ms/u/s!AiWjZ1LamlxzaTqcKb9QJrIZS7Y)|
| HRNetV2-W18 |26.2M|159.1| 1x |Y|N| 37.2 | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [FasterR-CNN-HR18-SyncBN-1x.pth](https://1drv.ms/u/s!Avk3cZ0cr1JedwR-inENTWU8X2E?e=llP4sR)|
| HRNetV2-W18 |26.2M|159.1| 1x |Y|Y(Default)| **37.6** | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [FasterR-CNN-HR18-SyncBN-MStrain-1x.pth](https://1drv.ms/u/s!Ao8vsd6OusckbJjMoiThi4DojsY?e=9qS2Mh)|
| HRNetV2-W18 |26.2M|159.1| 1x |Y|Y(ResizeCrop)| **37.6** | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) ||
| HRNetV2-W18 |26.2M|159.1| 2x |N|N| 38.1 | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [FasterR-CNN-HR18-2x.pth](https://1drv.ms/u/s!AiWjZ1LamlxzcHt7xyWTgVxmMLw)|
| HRNetV2-W18 |26.2M|159.1| 2x |Y|Y(Default)| **39.4** | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [FasterR-CNN-HR18-SyncBN-MStrain-2x.pth](https://1drv.ms/u/s!Ao8vsd6OusckbYWE2UwMf5fas7A?e=oi0lmh)|
| HRNetV2-W18 |26.2M|159.1| 2x |Y|Y(ResizeCrop)| **39.7** | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) ||
| HRNetV2-W32 |45.0M|245.3| 1x |N|N| 39.5 | [HRNetV2-W32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [FasterR-CNN-HR32-1x.pth](https://1drv.ms/u/s!AiWjZ1LamlxzaxRamJewuDqSozQ)|
| HRNetV2-W32 |45.0M|245.3| 1x |Y|Y(Default)| **41.0** | [HRNetV2-W32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [FasterR-CNN-HR32-SyncBN-MStrain-1x.pth](https://1drv.ms/u/s!Ao8vsd6Ousckab_Y65bdvmP9Qjk?e=LvWihi)|
| HRNetV2-W32 |45.0M|245.3| 2x |N|N| 40.8 | [HRNetV2-W32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [FasterR-CNN-HR32-2x.pth](https://1drv.ms/u/s!AiWjZ1LamlxzbE6rbdU9whYJkqs)|
| HRNetV2-W32 |45.0M|245.3| 2x |Y|Y(Default)| **42.6** | [HRNetV2-W32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [FasterR-CNN-HR32-SyncBN-MStrain-2x.pth](https://1drv.ms/u/s!Ao8vsd6OusckanYRdh_HXQRGFjQ?e=hBtvfo)|
| HRNetV2-W40 |60.5M|314.9| 1x |N|N| 40.4 | [HRNetV2-W40](https://1drv.ms/u/s!Aus8VCZ_C_33ck0gvo5jfoWBOPo) | [FasterR-CNN-HR40-1x.pth](https://1drv.ms/u/s!AiWjZ1LamlxzbE6rbdU9whYJkqs)|
| HRNetV2-W40 |60.5M|314.9| 2x |N|N| 41.4 | [HRNetV2-W40](https://1drv.ms/u/s!Aus8VCZ_C_33ck0gvo5jfoWBOPo) | [FasterR-CNN-HR40-2x.pth](https://1drv.ms/u/s!AiWjZ1Lamlxzb1Uy6QLZnsyfuFc)|

### Mask R-CNN

|Backbone|lr sched|Mask mAP|Box mAP|pretrained model|detection model|
|:--:|:--:|:--:|:--:|:--:|:--:|
| HRNetV2-W18 | 1x | 34.2 | 37.3 | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [MaskR-CNN-HR18-1x.pth](https://1drv.ms/u/s!AiWjZ1Lamlxzcfh06SXd2GR1zKw)|
| HRNetV2-W18 | 2x | 35.7 | 39.2 | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [MaskR-CNN-HR18-2x.pth](https://1drv.ms/u/s!AjfnYvdHLH5TafSZNlgq6UWnJWk)|
| HRNetV2-W32 | 1x | 36.8 | 40.7 | [HRNetV2-W32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [MaskR-CNN-HR32-1x.pth](https://1drv.ms/u/s!AiWjZ1LamlxzcugO3KlXfy_YhiE)|
| HRNetV2-W32 | 2x | 37.6 | 42.1 | [HRNetV2-W32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [MaskR-CNN-HR32-2x.pth](https://1drv.ms/u/s!AjfnYvdHLH5Taqt21comOmTbdBg)|


### Cascade R-CNN
**Note:** we follow the original paper[2] and adopt 280k training iterations which is equal to 20 epochs in mmdetection.

|Backbone|lr sched|mAP|pretrained model|detection model|
|:--:|:--:|:--:|:--:|:--:|
| ResNet-101  | 20e | 42.8 | [ResNet-101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) | [CascadeR-CNN-R101-20e.pth](https://1drv.ms/u/s!AiWjZ1LamlxzbvOFlCnGhXhKmsY)|
| HRNetV2-W32 | 20e | 43.7 | [HRNetV2-W32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [CascadeR-CNN-HR32-20e.pth](https://1drv.ms/u/s!AiWjZ1LamlxzasFUt8GWHW1Og3I)|


## Techniques about multi-scale training

#### Default

* Procedure 
    1. Select one scale from provided scales randomly and apply it.
    2. Pad all images in a GPU Batch(e.g. 2 images per GPU) to the same size (see `pad_size`, `1600*1000` or `1000*1600`)
    
* Code

You need to change lines below in config files

````python
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=8,
    pad_size=(1600, 1024),
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/train2017.zip',
        img_scale=[(1600, 1000), (1000, 600), (1333, 800)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True),
````


#### ResizeCrop

Less memory and less time, this implementation is more efficient compared to the former one

* Procedure
    
    1. Select one scale from provided scales randomly and apply it.
    2. Crop images to a fixed size randomly if they are larger than the given size.
    3. Pad all images to the same size (see `pad_size`).

* Code 

You need to change lines below in config files

````python
    imgs_per_gpu=2,
    workers_per_gpu=4,
    pad_size=(1216, 800),
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017.zip',
        img_scale=(1200, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=1,
        extra_aug=dict(
            rand_resize_crop=dict(
                scales=[[1400, 600], [1400, 800], [1400, 1000]],
                size=[1200, 800]
            )),
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True),
````



## Quick start
#### Environment
This code is developed using on Python 3.6 and PyTorch 1.0.0 on Ubuntu 16.04 with NVIDIA GPUs. Training and testing are 
performed using 4 NVIDIA P100 GPUs with CUDA 9.0 and cuDNN 7.0. Other platforms or GPUs are not fully tested.

#### Install
1. Install PyTorch 1.0 following the [official instructions](https://pytorch.org/)
2. Install `mmcv`
````bash
pip install mmcv
````
3. Install `pycocotools`
````bash
git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install \
 && cd ../../
````

4. Install `NVIDIA/apex` to enable **SyncBN**
````bash
git clone https://github.com/NVIDIA/apex
cd apex
python setup install --cuda_ext
````


5. Install `HRNet-Object-Detection`
````bash
git clone https://github.com/HRNet/HRNet-Object-Detection.git

cd HRNet-Object-Detection
# compile CUDA extensions.
chmod +x compile.sh
./compile.sh

# run setup
python setup.py install 

# or install locally
python setup.py install --user
````
For more details, see [INSTALL.md](INSTALL.md)

#### HRNetV2 pretrained models
```bash
cd HRNet-Object-Detection
# Download pretrained models into this folder
mkdir hrnetv2_pretrained
```
#### Datasets
Please download the COCO dataset from [cocodataset](http://cocodataset.org/#download). If you use `zip` format, please specify `CocoZipDataset` in **config** files or `CocoDataset` if you unzip the downloaded dataset. 

#### Train (multi-gpu training)
Please specify the configuration file in `configs` (learning rate should be adjusted when the number of GPUs is changed).
````bash
python -m torch.distributed.launch --nproc_per_node <GPUS NUM> tools/train.py <CONFIG-FILE> --launcher pytorch
# example:
python -m torch.distributed.launch --nproc_per_node 4 tools/train.py configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py --launcher pytorch
````

#### Test
````bash
python tools/test.py <CONFIG-FILE> <MODEL WEIGHT> --gpus <GPUS NUM> --eval bbox --out result.pkl
# example:
python tools/test.py configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py work_dirs/faster_rcnn_hrnetv2p_w18_1x/model_final.pth --gpus 4 --eval bbox --out result.pkl
````

**NOTE:** If you meet some problems, you may find a solution in [issues of official mmdetection repo](https://github.com/open-mmlab/mmdetection/issues) 
 or submit a new issue in our repo.
 
## Other applications of HRNets (codes and models):
* [Human pose estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
* [Semantic segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)
* [Facial landmark detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)
* [Image classification](https://github.com/HRNet/HRNet-Image-Classification)
 
## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{SunZJCXLMWLW19,
  title={High-Resolution Representations for Labeling Pixels and Regions},
  author={Ke Sun and Yang Zhao and Borui Jiang and Tianheng Cheng and Bin Xiao 
  and Dong Liu and Yadong Mu and Xinggang Wang and Wenyu Liu and Jingdong Wang},
  journal   = {CoRR},
  volume    = {abs/1904.04514},
  year={2019}
}
````

## Reference
[1] Deep High-Resolution Representation Learning for Human Pose Estimation. Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. CVPR 2019. [download](https://arxiv.org/pdf/1902.09212.pdf)

[2] Cascade R-CNN: Delving into High Quality Object Detection. Zhaowei Cai, and Nuno Vasconcetos. CVPR 2018.

## Acknowledgement
Thanks [@open-mmlab](https://github.com/open-mmlab) for providing the easily-used code and kind help!
