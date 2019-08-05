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
* [BingChen ZHAO](mailto:zhaobc@tongji.edu.cn)
