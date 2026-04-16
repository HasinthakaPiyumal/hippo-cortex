# `scripts/`

One-off utilities that aren't part of the importable `hippocortex` package:

- Dataset download / extraction (Split-CIFAR100, ImageNet-R).
- Preprocessing (resize, normalise, cache pickles).
- Evaluation sweeps across runs under `../results/`.
- Rosbag conversion for Stage-2 deployment logs.

Each script should be runnable standalone (`python scripts/<name>.py --help`) and self-document its arguments.
