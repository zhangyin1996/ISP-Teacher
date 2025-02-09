# Updates
## **2025/02/09: The code of our extended version 'ISP Dynamic Teacher' can be found in [[ISP Dynamic Teacher](https://github.com/zhangyin1996/ISP-Dynamic-Teacher)]**
+ 2024/03/29: Update environment requirements and label json file.
+ 2024/03/28: Code is available now.

# ISP-Teacher
[AAAI24] Official Pytorch Code for **ISP-Teacher: Image Signal Process with Disentanglement Regularization for Unsupervised Domain Adaptive Dark Object Detection**

Paper is available! [[Paper Link](https://ojs.aaai.org/index.php/AAAI/article/view/28569)]

![image text](https://github.com/zhangyin1996/ISP-Teacher/blob/main/pipeline.png "Pipeline")

## Abstract
Object detection in dark conditions has always been a great challenge due to the complex formation process of low-light images. Currently, the mainstream methods usually adopt domain adaptation with Teacher-Student architecture to solve the dark object detection problem, and they imitate the dark conditions by using non-learnable data augmentation strategies on the annotated source daytime images. Note that these methods neglected to model the intrinsic imaging process, i.e. image signal processing (ISP), which is important for camera sensors to generate low-light images. To solve the above problems, in this paper, we propose a novel method named ISP-Teacher for dark object detection by exploring Teacher-Student architecture from a new perspective (i.e. self-supervised learning based ISP degradation). Specifically, we first design a day-to-night transformation module that consistent with the ISP pipeline of the camera sensors (ISP-DTM) to make the augmented images look more in line with the natural low-light images captured by cameras, and the ISP-related parameters are learned in a self-supervised manner. Moreover, to avoid the conflict between the ISP degradation and detection tasks in a shared encoder, we propose a disentanglement regularization (DR) that minimizes the absolute value of cosine similarity to disentangle two tasks and push two gradients vectors as orthogonal as possible. Extensive experiments conducted on two benchmarks show the effectiveness of our method in dark object detection. In particular, ISP-Teacher achieves an improvement of +2.4% AP and +3.3% AP over the SOTA method on BDD100k and SHIFT datasets, respectively.

## 1. Environment 
+ Detectron2==0.6  [[Install Link](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)]  **Important !!!**
+ Install the appropriate versions of PyTorch and torchvision for your machine.
+ **In our setting:** 
`Cuda==10.2, Python==3.8, Pytorch==1.10.1, Detectron2==0.6`

## 2. Datasets
+ Download the BDD100k or SHIFT datasets.
+ Split dataset into two parts using labels ‘day’ and ‘night’. Convert datasets labels to coco format. You can download split json file from [Baiduyun](https://pan.baidu.com/s/1lExTex7JjZ9-4DZ_fWciSg?pwd=1234)(password:1234) or [Google Drive](https://drive.google.com/drive/folders/1ynapIcAm5subozk0QNzrPWHwWj_xpGle?usp=drive_link). Please refer to [2PCNet](https://github.com/mecarill/2pcnet) (CVPR2023) for more details.
+ Replace the dataset and label path in `twophase/data/datasets/builtin.py #188~#212` with you own.


## 3. Train
+ For BDD100k as an example, the command for training ISP-Teacher on 4 RTX6000 GPUs is as following:
```
python train_net.py --num-gpus 4 --config configs/faster_rcnn_R50_bdd100k.yaml OUTPUT_DIR output/bdd100k
```

## 4. Evaluation
+ For BDD100k as an example, you could use your trained model or use ours pretrained model from: [Baiduyun](https://pan.baidu.com/s/1vYYKX9BdlQIHY-Y7E9n5nA?pwd=1234)(password:1234) or [Google Drive](https://drive.google.com/file/d/1qCXQkiJ3fAHtmKi6U0CnXTzgiGKRfxIC/view?usp=drive_link).
+ The command for evaluating ISP-Teacher on one RTX3090 GPU is as following:
```
python train_net.py --eval-only --config configs/faster_rcnn_R50_bdd100k.yaml MODEL.WEIGHTS <your weight>.pth
```

## 5. Citation
If you find ISP-Teacher useful in your research, please consider citing:
```
@inproceedings{zhang2024isp,
  title={ISP-Teacher: Image Signal Process with Disentanglement Regularization for Unsupervised Domain Adaptive Dark Object Detection},
  author={Zhang, Yin and Zhang, Yongqiang and Zhang, Zian and Zhang, Man and Tian, Rui and Ding, Mingli},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7387--7395},
  year={2024}
}
```

## 6. Acknowledgements
+ The code is based on [Detectron2](https://github.com/facebookresearch/detectron2) and [2PCNet](https://github.com/mecarill/2pcnet) (CVPR2023).
+ In addition, some codes are borrowed from [MAET](https://github.com/cuiziteng/ICCV_MAET) (ICCV2021).

**Many thanks for these great works!**


