# CVWC-2019
ICCV 2019 Workshop &amp; Challenge on Computer Vision for Wildlife Conservation (CVWC) Detection Track

# Methods

## Data Pre-processing
Before we designed our model, we looked into the dataset carefully and the details are as follows:

The distribution of bounding box according to size, count and square.

![Count](https://github.com/ElegantLin/CVWC-2019/blob/master/doc/count.png)
![Square](https://github.com/ElegantLin/CVWC-2019/blob/master/doc/square.png)
![Width](https://github.com/ElegantLin/CVWC-2019/blob/master/doc/width.png)
![Height](https://github.com/ElegantLin/CVWC-2019/blob/master/doc/height.png)


## One-stage Method
We performed SSD on 1 GPU and the result is as follows:

## Two-stage Method
We tried faster-RCNN with HRNet as the backbone and the result is as follows:

## Three-stage Method
To get a better performance, we tried Cascade-RCNN with HRNet as the backbone 

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
