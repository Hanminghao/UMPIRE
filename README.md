# Towards Unified Molecule-Enhanced Pathology Image Representation Learning via Integrating Spatial Transcriptomics

***TL;DR:*** UMPIRE introduces the first large-scale multimodal pre-training framework for pathology images and spatial transcriptomics.

## Installation
First clone the repo and cd into the directory:
```shell
git clone https://github.com/Hanminghao/UMPIRE.git
cd UMPIRE
```
Then create a conda env and install the dependencies:
```shell
conda create -n umpire python=3.9 -y
conda activate umpire
pip install --upgrade pip
pip install -e .
```

## Updates

- **03/04/2025**: Updated the tokenize step and inference step in the downstream task.

- **03/04/2025**: Updated the pre-trained weights with [CONCH](https://github.com/mahmoodlab/CONCH), [Phikon](https://huggingface.co/owkin/phikon), and [UNI](https://github.com/mahmoodlab/UNI).

## Preparing and loading the model
1. Request access to the model weights and example data from [Google Drive](https://drive.google.com/drive/folders/1K8GxOEgBwzIitUXLKQaPSmKTf-XLTZx2?usp=sharing).

2. Run `tokenize_downstream.py` to tokenize the downstream data.  

3. Run `tutorial.ipynb` to learn downstream encoding and t-SNE visualization.


## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://arxiv.org/abs/2412.00651):

Han, M., Yang, D., Cheng, J., Zhang, X., Qu, L., Chen, Z., & Zhang, L. (2024). Towards Unified Molecule-Enhanced Pathology Image Representation Learning via Integrating Spatial Transcriptomics. arXiv preprint arXiv:2412.00651.



```
@article{han2024towards,
  title={Towards Unified Molecule-Enhanced Pathology Image Representation Learning via Integrating Spatial Transcriptomics},
  author={Han, Minghao and Yang, Dingkang and Cheng, Jiabei and Zhang, Xukun and Qu, Linhao and Chen, Zizhi and Zhang, Lihua},
  journal={arXiv preprint arXiv:2412.00651},
  year={2024}
}
```