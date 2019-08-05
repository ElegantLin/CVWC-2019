# CVWC-2019
ICCV 2019 Workshop &amp; Challenge on Computer Vision for Wildlife Conservation (CVWC) Detection Tracks

## One-stage Method
We performed SSD on 1 GPU and the result is as follows:

## Two-stage Method
We tried faster-RCNN with HRNet as the backbone on 8 GPUs. The structure of HRNet is: 
![HRNet For Object Detection](https://github.com/HRNet/HRNet-Object-Detection/blob/master/images/hrnetv2p.png)

## Three-stage Method
To get a better performance, we tried Cascade-RCNN with HRNet as the backbone. The structure of HRNet is the same as the plot above. The structure of Cascade RCNN is
![Cascade RCNN](https://github.com/ElegantLin/CVWC-2019/blob/master/doc/cascade.png)

## Post-processing

## Experiment and Results

|        Method         | mAP  | FLOP | Param | Sync BN | Models |
| :-------------------: | :--: | :--: | :---: | :-----: | :----: |
|        SSD-512        |      |      |       |    N    |        |
| Faster RCNN （HRNet） |      |      |       |    Y    |        |
| Cascade RCNN（HRNet） |      |      |       |    N    |        |

##Other Experiments

|     Methods     | mAP  |  △   | Overall |
| :-------------: | :--: | :--: | :-----: |
|   Faster RCNN   |      |      |         |
| + anchor resize |      |      |         |
| + cosine decay  |      |      |         |
|    + mix up     |  ?   |  ?   |    ?    |

? indicate the experiments are not finished due to the limited time. If we are invited to write a paper, we will fulfill all the experiments.

## Setup

### Train

### Inference

## Reference

1. [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)
2. [HRNetV2 for object detection, semantic segmentation, facial landmark detection, and image classification](https://arxiv.org/pdf/1904.04514.pdf)
3. [MMDetection: Open MMLab Detection Toolbox and Benchmark](https://arxiv.org/pdf/1906.07155.pdf)
4. [Soft-NMS -- Improving Object Detection With One Line of Code](https://arxiv.org/pdf/1704.04503.pdf)
5. [Cosine Learning Rate Decay](https://arxiv.org/pdf/1806.01593.pdf)