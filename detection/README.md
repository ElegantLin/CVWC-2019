# CVWC-2019
ICCV 2019 Workshop &amp; Challenge on Computer Vision for Wildlife Conservation (CVWC) Detection Track.

## One-stage Method
We performed SSD on 1 GPU.

## Two-stage Method
We tried faster-RCNN with HRNet as the backbone on 8 GPUs. The structure of HRNet is: 
![HRNet For Object Detection](https://github.com/HRNet/HRNet-Object-Detection/blob/master/images/hrnetv2p.png)

## Three-stage Method
To get a better performance, we tried Cascade-RCNN with HRNet as the backbone on 8 GPUs. The structure of HRNet is the same as the plot above. The structure of Cascade RCNN is
![Cascade RCNN](https://github.com/ElegantLin/CVWC-2019/blob/master/doc/cascade.png)

## Post-processing

## Experiment and Results

We trained SSD-512 for 600 epochs, Faster RCNN for 30 epochs and Cascade RCNN for 30 epochs.

|                   Method                    |   mAP   | GFLOP | #Param | Sync BN | lr sched | Pre-Trained | Model | log  |
| :-----------------------------------------: | :-----: | :---: | :----: | :-----: | :------: | :---------: | :---: | ---- |
|                   SSD-512                   | 0.47114 |       |        |    N    |    *     |             |       |      |
| Faster RCNN （HRNet, multi-scale training） | 0.60001 | 245.3 | 45.0M  |    Y    |    2x    |             |       |      |
|            Cascade RCNN（HRNet）            | 0.57970 |       |        |    N    |   20e    |             |       |      |

## Other Experiments 

|              Methods               | mAP  |  △   | Overall |
| :--------------------------------: | :--: | :--: | :-----: |
| Faster RCNN (Multi-scale training) |      |      |         |
|          + anchor resize           |      |      |         |
|           + cosine decay           |      |      |         |
|              + mix up              |  ?   |  ?   |    ?    |

|       Methods        | mAP  |  △   | Overall |
| :------------------: | :--: | :--: | :-----: |
| Cascade RCNN (HRNet) |      |      |         |
|   + anchor resize    |  ？  |  ？  |   ？    |
|    + cosine decay    |      |      |         |
|       + mix up       |  ?   |  ?   |    ?    |

? indicates the experiments are not finished due to the limited time. If we are invited to write a paper, we will fulfill all the experiments.

## Setup

### Train (8 GPUs)

Set up the environment according to [HRNet Instruction]([https://github.com/HRNet/HRNet-Object-Detection#Quick%20Start](https://github.com/HRNet/HRNet-Object-Detection#Quick Start)). Take Faster RCNN as example,

```bash
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py configs/hrnet/faster_rcnn_hrnetv2p_w32_syncbn_mstrain_2x.py --launcher pytorch
```

### Inference (4 GPUs)

```bash
python tools/test.py configs/hrnet/faster_rcnn_hrnetv2p_w32_syncbn_mstrain_2x.py work_dirs/faster_rcnn_hrnetv2p_w32_2x/model_final.pth --gpus 4 --eval bbox --out result.pkl
```

## Reference

1. [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)
2. [HRNetV2 for object detection, semantic segmentation, facial landmark detection, and image classification](https://arxiv.org/pdf/1904.04514.pdf)
3. [MMDetection: Open MMLab Detection Toolbox and Benchmark](https://arxiv.org/pdf/1906.07155.pdf)
4. [Soft-NMS -- Improving Object Detection With One Line of Code](https://arxiv.org/pdf/1704.04503.pdf)
5. [Cosine Learning Rate Decay](https://arxiv.org/pdf/1806.01593.pdf)