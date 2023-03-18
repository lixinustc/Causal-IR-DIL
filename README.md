# Causal-IR: Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective
[Xin Li](http://home.ustc.edu.cn/~lixin666/), [Bingchen Li](), [Xin Jin](http://home.ustc.edu.cn/~jinxustc/), [Cuiling Lan](https://scholar.google.com/citations?user=XZugqiwAAAAJ&hl=en), [Zhibo Chen](https://scholar.google.com/citations?user=1ayDJfsAAAAJ&hl=en)

University of Science and Technology of China, Microsoft Research Asia (MSRA), Eastern Institute of Technology (EIT) 

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2303.06859.pdf)


![image](https://github.com/USTC-IMCL/SwinIQA/blob/main/fig/SwinIQA.png)

## Abstract
In recent years, we have witnessed the great advancement of Deep neural networks (DNNs) in image restoration. However, a critical limitation is that they cannot generalize well to real-world degradations with different degrees or types. In this paper, we are the first to propose a novel training strategy for image restoration from the causality perspective, to improve the generalization ability of DNNs for unknown degradations. Our method, termed Distortion Invariant representation Learning (DIL), treats each distortion type and degree as one specific confounder, and learns the distortion-invariant representation by eliminating the harmful confounding effect of each degradation. We derive our DIL with the back-door criterion in causality by modeling the interventions of different distortions from the optimization perspective. Particularly, we introduce counterfactual distortion augmentation to simulate the virtual distortion types and degrees as the confounders. Then, we instantiate the intervention of each distortion with a virtual model updating based on corresponding distorted images, and eliminate them from the meta-learning perspective. Extensive experiments demonstrate the effectiveness of our DIL on the generalization capability for unseen distortion types and degrees.

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
