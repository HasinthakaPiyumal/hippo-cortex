# `experiments/`

One subfolder per experiment. Each subfolder is self-contained: config file(s), any glue scripts, a `README.md` describing the setup, and a pointer to where outputs land under `../results/<experiment>/`.

## Subfolders

| Folder                        | Stage | Purpose                                                                    |
| ----------------------------- | ----- | -------------------------------------------------------------------------- |
| `stage1_split_cifar100/`      | 1     | Split-CIFAR100 (20 tasks) benchmark vs. Mamba-CL and Inf-SSM baselines.    |
| `stage1_imagenet_r/`          | 1     | ImageNet-R continual-learning benchmark.                                   |
| `stage2a_hybrid_vision/`      | 2a    | Hybrid Mamba+Transformer training on vision datasets (no robot yet).       |
| `stage2b_sim_mujoco/`         | 2b    | Simulator validation in Mujoco / Isaac Sim Go2 environment.                |
| `stage2c_go2_deployment/`     | 2c    | Real Go2 Edu deployment at WSO2 Colombo; sequential task curriculum.       |

## Conventions

- Experiment folder names are `stage<N><letter?>_<slug>`.
- Outputs (checkpoints, metrics, plots) go to `../results/<experiment_folder>/<run_id>/`.
- Don't put dataset files here — they live under `../data/`.
