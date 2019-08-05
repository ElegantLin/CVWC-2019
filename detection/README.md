# CVWC-2019
ICCV 2019 Workshop &amp; Challenge on Computer Vision for Wildlife Conservation (CVWC) Detection Track.

## Method
We tried faster-RCNN with HRNet as the backbone on 8 GPUs. The structure of HRNet is: 
![HRNet For Object Detection](https://github.com/HRNet/HRNet-Object-Detection/blob/master/images/hrnetv2p.png)


## Experiment and Results

We trained Faster RCNN for 30 epochs. We used the best model according to the validation result. Epoch_25 has the best performance. You should first convert the Pascal VOC format to coco format using [voc2coco]() tools. And then use a fake testing result. Place it as

```
--{$YOUR DATA ROOT}
	--coco
		--annotations
			--instances_train2014.json
			--instances_val2014.json
			--instances_test2014.json
		--train2014
			--[train images]
		--val2014
			--[validation images]
		--test2014
			--[test images]
```

And config `data_root` it in your [config file](https://github.com/ElegantLin/CVWC-2019/blob/master/detection/HRNet-Object-Detection/configs/hrnet/faster_rcnn_hrnetv2p_w32_syncbn_mstrain_2x.py)

|                   Method                    |     mAP     | GFLOP | #Param | Sync BN | lr sched |                         pre-trained                          |                       detection model                        |           log           |                            config                            |                            result                            |
| :-----------------------------------------: | :---------: | :---: | :----: | :-----: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Faster RCNN （HRNet, multi-scale training） | **0.60093** | 245.3 | 45.0M  |    Y    |    2x    | [HRNet_w32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [epoch_25.pth](https://drive.google.com/open?id=1nFtlXrhfRX7Yi2PMgPgniBAn-MN4js9X) | [20190801_010901.log]() | [faster_rcnn_hrnetv2p_w32_syncbn_mstrain_2x.py](https://github.com/ElegantLin/CVWC-2019/blob/master/detection/HRNet-Object-Detection/configs/hrnet/faster_rcnn_hrnetv2p_w32_syncbn_mstrain_2x.py) | [results.json](https://github.com/ElegantLin/CVWC-2019/blob/master/detection/result.json) |

## Other Experiments 

|              Methods               |     mAP     |    △     | Overall  |
| :--------------------------------: | :---------: | :------: | :------: |
| Faster RCNN (Multi-scale training) | **0.60093** |   0.00   |   0.00   |
|    + anchor resize (24, 32, 40)    |   0.58244   | -0.01849 | -0.01849 |
|           + cosine decay           |   0.57452   | -0.00792 | -0.02641 |
|              + mix up              |      ?      |    ?     |    ?     |

? indicates the experiments are not finished due to the limited time. If we are invited to write a paper, we will fulfill all the experiments.

## Setup

### Train (8 GPUs)

Set up the environment according to [HRNet Instruction](https://github.com/HRNet/HRNet-Object-Detection#Quick%20Start) . Take Faster RCNN as example,

```bash
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py configs/hrnet/faster_rcnn_hrnetv2p_w32_syncbn_mstrain_2x.py --launcher pytorch
```

### Inference (4 GPUs)

Please download the weight first and put it into `{$Your working directory}/work_dirs/faster_rcnn_hrnetv2p_w32_syncbn_mstrain_2x/`

```bash
python tools/test.py configs/hrnet/faster_rcnn_hrnetv2p_w32_syncbn_mstrain_2x.py work_dirs/faster_rcnn_hrnetv2p_w32_2x_2/epoch_25.pth --gpus 4 --eval bbox --out result.pkl
```

The file you get does not fit `coco` format. Please use `convert.py` to convert it.

```bash
python convert.py [origin json] [target json]
```



## Reference

1. [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf)
2. [HRNetV2 for object detection, semantic segmentation, facial landmark detection, and image classification](https://arxiv.org/pdf/1904.04514.pdf)
3. [MMDetection: Open MMLab Detection Toolbox and Benchmark](https://arxiv.org/pdf/1906.07155.pdf)
4. [Soft-NMS -- Improving Object Detection With One Line of Code](https://arxiv.org/pdf/1704.04503.pdf)
5. [Cosine Learning Rate Decay](https://arxiv.org/pdf/1806.01593.pdf)
6. [HRNet-Object-detection](https://github.com/HRNet/HRNet-Object-Detection.git)