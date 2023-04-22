# Causal-IR: Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective
[Xin Li](http://home.ustc.edu.cn/~lixin666/), [Bingchen Li](), [Xin Jin](http://home.ustc.edu.cn/~jinxustc/), [Cuiling Lan](https://scholar.google.com/citations?user=XZugqiwAAAAJ&hl=en), [Zhibo Chen](https://scholar.google.com/citations?user=1ayDJfsAAAAJ&hl=en)

University of Science and Technology of China (USTC), Microsoft Research Asia (MSRA), Eastern Institute of Technology (EIT) 

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2303.06859.pdf)


![image](https://github.com/lixinustc/Casual-IR-DIL/blob/main/figs/visualization.png)

## Abstract
In recent years, we have witnessed the great advancement of Deep neural networks (DNNs) in image restoration. However, a critical limitation is that they cannot generalize well to real-world degradations with different degrees or types. In this paper, we are the first to propose a novel training strategy for image restoration from the causality perspective, to improve the generalization ability of DNNs for unknown degradations. Our method, termed Distortion Invariant representation Learning (DIL), treats each distortion type and degree as one specific confounder, and learns the distortion-invariant representation by eliminating the harmful confounding effect of each degradation. We derive our DIL with the back-door criterion in causality by modeling the interventions of different distortions from the optimization perspective. Particularly, we introduce counterfactual distortion augmentation to simulate the virtual distortion types and degrees as the confounders. Then, we instantiate the intervention of each distortion with a virtual model updating based on corresponding distorted images, and eliminate them from the meta-learning perspective. Extensive experiments demonstrate the effectiveness of our DIL on the generalization capability for unseen distortion types and degrees.

## New!!!
| 2023-04-21  | Release the code for one version **(serial first-order) (cheap training)** of **Distortion-invariant Learning (DIL)** on **image denosing** | 

More version will be released progressively.

Six tasks: Image denoising, Image deblurring, hybrid-distorted IR, Real Image Denosing, Real Image Super-resolution, Image Deraining

## Getting Start

### Clone repo
```bash
git clone https://github.com/lixinustc/Causal-IR-DIL.git
cd Causal-IR-DIL
```

### Prepare environment
```bash
conda create -n DIL python=3.8
conda activate DIL
pip install -r requirements.txt
```
Our codes are compatible with pytorch1.9.0, you may try newer version.

### Prepare training dataset
Download 800 DIV2K and 2650 Flickr2K training images from [this link](https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW?usp=sharing) (Google drive).
To accelerate I/O speed, we firstly crop these images into 256x256 patches. To do so, please first run
```
python generate_cropped_DF2K.py
```
Remember to replace ''\<path to your downloaded DF2K dataset>'' and ''\<path to your output cropped training dataset>'' according to your preference. The number of cropped patches should be 118101.

### Prepare testing dataset
You may download commonly used testing datasets following [this link](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u).

### Training

- Single GPU training
```bash
CUDA_VISIBLE_DEVICES=0 python DIL_sr_noise.py --ckpt_save <path to save your checkpoints> --trainset <path to your cropped DF2K> --batch_size 8 
```

- Distributed training (4 GPUs as an example)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 DIL_sr_noise.py --ckpt_save <path to save your checkpoints> --trainset <path to your cropped DF2K> --batch_size 8 --gpus 4 --distributed
```

Please refer to code for more information.

### Testing
```bash
python eval_noise.py --ckpt <path to your checkpoint> --testset <path to your testset> --save <path to save results> --level <gaussian noise level>
```
As for "level", you may try distortion levels used in training (5, 10, 15, 20), or distortion levels that are unseen during training (where DIL shows its strength!).




## Cite US
Please cite us if this work is helpful to you.


```
@article{li2023learning,
  title={Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective},
  author={Li, Xin and Li, Bingchen and Jin, Xin and Lan, Cuiling and Chen, Zhibo},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```


---
