# HippoCortex

**Biologically-Inspired Continual Learning for Edge Robotics via Sharp Wave Ripple Generative Replay in State Space Models**

HippoCortex translates the mammalian two-system memory model (hippocampus + neocortex) into a trainable continual-learning framework. Rather than buffering raw past data, it stores only compact statistics `(µ, σ²)` of Mamba hidden states and regenerates synthetic memory traces during consolidation via a conditional-VAE SWR generator and a null-space gradient projector `P = I − UUᵀ`.

The project runs in two stages:

- **Stage 1 (Months 1–3)** — Core algorithm on a Mamba SSM backbone; validation on Split-CIFAR100 (20 tasks) and ImageNet-R. Target: +8–10% over Mamba-CL, +4% over Inf-SSM.
- **Stage 2 (Months 4–10)** — Hybrid Mamba + Transformer architecture; simulator validation (Mujoco / Isaac Sim); real-world deployment on the Unitree Go2 Edu (Jetson Orin NX 16GB, RealSense D435i, L1 LiDAR, IMU, ROS2).

Full proposal: [`proposal/HippoCortex-wso2_proposal.pdf`](proposal/HippoCortex-wso2_proposal.pdf).

## Team

| Role        | Name                | Contact                              |
| ----------- | ------------------- | ------------------------------------ |
| Student     | Hasinthaka Piyumal  | senanay-se21036@stu.kln.ac.lk        |
| Student     | Praveen Dedigama    | dedigam-se21031@stu.kln.ac.lk        |
| Student     | Induwara Mihisara   | mihisar-se21025@stu.kln.ac.lk        |
| Student     | Thagya Kavindi      | kavindi-se21062@stu.kln.ac.lk        |
| Supervisor  | Dr. Nalin Warnajith | nwarnajith@kln.ac.lk                 |

B.Sc. Hons. Software Engineering (UG), University of Kelaniya.

## Repository layout

```
hippocortex/        # main Python package (models, cl, data, training, robot, utils)
experiments/        # per-experiment configs and run scripts
lit-review/         # Obsidian vault — papers, topic notes, meetings, daily log
docs/               # architecture spec, hardware notes, timeline
papers/             # LaTeX manuscripts (stage 1 + stage 2)
proposal/           # original research proposal
notebooks/          # exploratory Jupyter notebooks
scripts/            # dataset download, preprocessing, eval utilities
tests/              # pytest unit tests
data/               # datasets — gitignored
checkpoints/        # model weights — gitignored
results/            # plots, tables, run logs — gitignored
```

## Quickstart

_To be filled in once Stage 1 code lands. Expected flow: create a virtualenv, install the package in editable mode, run a Split-CIFAR100 experiment from `experiments/stage1_split_cifar100/`._

## Literature review

The `lit-review/` folder is an [Obsidian](https://obsidian.md) vault. Open `lit-review/` as a vault in Obsidian to get backlinks and graph view across paper notes, topic notes, meeting minutes, and the daily research log.
