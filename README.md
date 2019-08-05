# CVWC-2019
ICCV 2019 Workshop &amp; Challenge on Computer Vision for Wildlife Conservation (CVWC) Detection and ReID Tracks

## Data Pre-processing
Before we designed our model, we looked into the dataset carefully and the details are as follows:

The distribution of bounding box according to size, count and square.

![Details](https://github.com/ElegantLin/CVWC-2019/blob/master/doc/Figure_1.png)

## Results

We joined 3 tracks of 4 in this workshop which are detection, plain ReID and wild ReID. 

The final results are:

|   Track    | mAP(single_cam in ReID) | top-1(single_cam) | top-5(single_cam) | mAP(cross_cam) | top-1(single_cam) | top-5(cross_cam) | FLOP | Param |
| :--------: | :---------------------: | :---------------: | :---------------: | :------------: | :---------------: | :--------------: | :--: | :---: |
| Detection  |          0.60           |         -         |         -         |       -        |         -         |        -         |      |       |
| Plain ReID |          0.84           |       0.95        |       0.99        |      0.46      |       0.87        |       0.93       |      |       |
| Wild ReID  |          0.82           |       0.93        |       0.96        |      0.46      |       0.84        |       0.90       |      |       |

Please refer to the respective folder for more details.

## Members
* [Zonglin DI](mailto:1452640@tongji.edu.cn)
* [BingChen Zhao](mailto:zhaobc@tongji.edu.cn)

## One-stage Method
We performed SSD on 1 GPU and the result is as follows:

## Two-stage Method
We tried faster-RCNN with HRNet as the backbone. The structure of HRNet is: 
![HRNet For Object Detection](https://github.com/HRNet/HRNet-Object-Detection/blob/master/images/hrnetv2p.png)

## Three-stage Method
To get a better performance, we tried Cascade-RCNN with HRNet as the backbone. The structure of HRNet is 

## Post-processing

# Experiment and Results

## SSD(VGG-512)
## SSD(VGG-300)

## Faster RCNN HRNet

## Cascade R-CNN HRNet

## Results Illustration

? indicate the experiments are not finished due to the limited time. If we are invited to write a paper, we will fulfill all the experiments.

# Reference

1. [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)
2. [HRNetV2 for object detection, semantic segmentation, facial landmark detection, and image classification](https://arxiv.org/pdf/1904.04514.pdf)
3. [MMDetection: Open MMLab Detection Toolbox and Benchmark](https://arxiv.org/pdf/1906.07155.pdf)
4. [Soft-nms]()
5. [Guided Anchoring]()
6. [Cosine Learning Rate Decay]()
