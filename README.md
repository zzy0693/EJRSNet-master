# EJRSNet

## Introduction

[EJRSNet:A Joint Mechanism Network for Object Detection with Efficient Receiving and Sending of Hierarchical Feature Information]
![](C:\Users\admin\Desktop\model.jpg)
Feature Pyramid Networks (FPNs) can integrate features at different scales layer-by-layer, thereby enhancing object detection performance. However, the lack of interaction for information fusion across inter-levels hinders the acquisition of contextual feature information and limits the further development of object detectors. Additionally, it is worth noting that the Convolutional Neural Network (CNN) lacks non-local information extraction, while Transformer is inadequate for local information extraction and optimization. To address these challenges, this paper proposes an object detector based on CNN and Transformer's Efficient Joint Mechanism for Receiving and Sending (EJRS) of Hierarchical Feature Information for encoding-decoding to enhance efficient interactive transmission of feature information at different hierarchical levels. Firstly, the encoder's Multi-level Feature Information Pre-enhanced Integration (MFIPI) module is designed to pre-enhance the model's feature extraction for various scales of spatial and semantic information, as well as to enhance dependencies between feature channels. Secondly, a Joint Efficient Transmission and Receipt Mechanism(JETM) architecture combining CNN and Transformer decoder is designed to facilitate cross-layer collection and sending of high-resolution spatial features at lower levels and semantic features at higher levels, thereby improving the model's ability for information fusion interaction. This also enhances both local and global extraction of feature information at different levels while establishing long-range dependencies between object pixels. Among them, the optimized decoder Transformer has obvious advantages over the original Transformer. Experimental results show accuracies of 47.4% on the COCO2017 dataset and 82.77% on the PASCAL VOC2007+2012 datasets respectively, validating the effectiveness of our proposed method.

## Prerequisites

Win10;
Python 3.8+;
PaddleDetection=2.8.0;
CUDA 11.2+ (If you build Paddle from source, CUDA 11.1 is also compatible);
Paddle=2.4.2;

## Model Zoo

| Backbone | Model | Images/GPU | Inf time (fps) | Box AP |   Config    | Download |
|:------:|:--------:|:----------:|:--------------:|:------:|:-----------:|:--------:|
| R-50 | EJRSNet  |     1      |     12.41      |  44.0  | [config](https://github.com/zzy0693/EJRSNet-master/blob/master/configs/EJRSNet/EJRS_r50_fpn_1x_coco.yml) | [model](https://drive.google.com/drive/folders/1XOw8pOEM1jEU87XWjjlmArn92XqFVTVg?usp=sharing) |

## Run command

The run commands still follow the PaddleDetection library command format. For example, the training model configuration command: python train.py -c configs/tood/tood_r50_fpn_1x_coco.yml --eval.

**Notes:**

- EJRSNet is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.
- EJRSNet uses GPU-V100 to train 12 epochs or 24 epochs.

GPU single-card training

```
Acknowledgement
The implementation of EJRSNet is based on PaddleDetection.

License
This project is released under the Apache 2.0 license.
```
## Citations
@inproceedings{zhang2024ejrsnet,
    title={EJRSNet:A Joint Mechanism Network for Object Detection with Efficient Receiving and Sending of Hierarchical Feature Information},
    author={Zhang, Zhenyi and Zhu, Mengdi and Yang, Xiaolong and  Li, Tianping*},
    booktitle={ESWA},
    year={2024}
}
```
