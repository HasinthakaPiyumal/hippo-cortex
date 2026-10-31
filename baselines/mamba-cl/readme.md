# Mamba-CL: Optimizing Selective State Space Model in Null Space for Continual Learning

## Environment

- GPU: NVIDIA GeForce RTX 4090
- Python: 3.11.5

```
torch==2.2.2
torchvision==0.17.2
timm==1.0.9
einops==0.8.0
numpy==1.26.4
opencv-python==4.10.0.84
Pillow
regex
scikit-image
scikit-learn
scipy
tqdm
```
Please refer to https://github.com/OpenGVLab/De-focus-Attention-Networks to install Mamba.

Download the pre-training weights: "https://github.com/OpenGVLab/De-focus-Attention-Networks/releases/download/v1.0/defocus_mamba_large_cls_21k.pth"


## Dataset preparation
### 1. Download the datasets and uncompress them:

- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- ImageNet-R: https://github.com/hendrycks/imagenet-r
- DomainNet: https://ai.bu.edu/M3SDA/

### 2. Rearrange the directory structure:

We use a unified directory structure for all datasets:
```
DATA_ROOT
    |- train
    |    |- class_folder_1
    |    |    |- image_file_1
    |    |    |- image_file_2
    |    |- class_folder_2
    |         |- image_file_2
    |         |- image_file_3
    |- val
         |- class_folder_1
         |    |- image_file_5
         |    |- image_file_6
         |- class_folder_2
              |- image_file_7
              |- image_file_8
```
We provide the scripts `split_[dataset].py` in the `tools` folder to rearange the directory structure.
Please change the `root_dir` in each script to the path of the uncompressed dataset.

## Training and evaluation

10-split ImageNet-R: `train_imagenet_r.sh`

20-split ImageNet-R: `train_imagenet_r-s20.sh`

10-split CIFAR-100: `train_cifar100.sh`

10-split DomainNet: `train_idomainnet.sh`



Please specify the `--data_root` argument in the above bash scripts to the locations of the datasets.
Change the `--seed` argument to use different seeds (e.g., 2025, 2026).

## Vim-small backbone (HippoCortex addition)

`-m vim_small` runs Mamba-CL on Vision Mamba small, the backbone of our Inf-SSM baseline
(`utils/vim_mamba.py`, same network and parameter names as `baselines/inf-ssm/mamba_vim.py`).
`--pretrained_path vim_small_in1k` downloads the hustvl ImageNet-1K weights from Hugging Face; a file path also works.

- Vim is bidirectional, so each block has 9 null-space parameters: `x_proj_{step,B,C}`, `A_log` for each
  direction, plus `out_proj.weight`. The backward A is stored as `A_b_log`, so `A_b_log` is added to
  `--training_string` and `--null_patterns` automatically.
- `delta` is computed outside the scan kernel (`delta_softplus=False`), as in the De-focus mixer, so the
  `A` and `B` readers see the real step size.
- `--grad_ckpt True` recomputes each block in backward if a batch does not fit.

