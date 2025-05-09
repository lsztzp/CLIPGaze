# CLIPGaze: Zero-Shot Goal-Directed scanpath prediction using CLIP

Goal-directed scanpath prediction aims to predict people's gaze shift path when searching for objects in a visual scene. Most existing goal-directed scanpath prediction methods cannot generalize to target classes not present during training. Besides, they usually exploit different pre-trained models to extract features for the target prompt and image, resulting in big feature gap and making the subsequent feature matching and fusion very difficult. To solve the above problems, we propose a novel zero-shot goal-directed scanpath prediction model named CLIPGaze. We use CLIP to extract pre-matched features for the target prompt and input image, making the feature fusion easier to receive. Using large model like CLIP can also enhance the whole model's generalization ability on target classes not present during training. We propose a hierarchical visual-semantic feature fusion module to fuse the target and image features more comprehensively. Furthermore, due to the limited number of classes in goal-directed scanpath dataset, we employ image segmentation as a proxy task to help train the feature fusion module, significantly enhancing our model's performance in zero-shot setting.

# Installation

```bash
conda create -n  CLIPGaze python=3.8.5
conda activate CLIPGaze
bash install.sh
```

# Files to download
We suggest that you first reproduce [Gazeformer](https://arxiv.org/abs/2303.15274) from this [repository](https://github.com/cvlab-stonybrook/Gazeformer). Here, you can find the training files related to COCO_Search18 (e.g., './dataset/coco_search18_TP_Train.json') and the corresponding test files (e.g., './SemSS/test_TP_Sem.pkl').

You can download the traditional target present model weight from this [link](https://drive.google.com/drive/folders/1CO7OSwEy8dn3xPkexi3ZfcaAOOS_ecpr?usp=sharing).

# Scripts

First Step: Extract the related CLIP visual and textual feature, run the following
```bash
python feature_extrctor.py
```
To train the model, run the following
```bash
python train.py
```
To evaluation the model, run the following
```bash
python test.py
```

Acknowledgement - A portion of our code is adapted from this [repository](https://github.com/cvlab-stonybrook/Gazeformer) for [Gazeformer](https://arxiv.org/abs/2303.15274) model. We would like to thank the authors Mondal et al. for open-sourcing their code. 

# Citation
If you use this work, please cite as follows :
```
@inproceedings{lai2025clipgaze,
  title={CLIPGaze: Zero-Shot Goal-Directed Scanpath Prediction Using CLIP},
  author={Lai, Yantao and Quan, Rong and Liang, Dong and Qin, Jie},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
