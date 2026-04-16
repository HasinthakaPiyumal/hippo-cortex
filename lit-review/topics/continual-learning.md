---
tags: [topic, continual-learning]
---

# Continual learning

Training a single model on a **sequential stream of tasks** without erasing what it already knows. The failure mode when you do nothing is **catastrophic forgetting** ([[papers/kirkpatrick2017-ewc]]).

## Families of methods

- **Replay-based** — store raw past samples in a buffer, interleave during new-task training. Memory-heavy, privacy-unsafe. Example: experience replay.
- **Generative replay** — train a generator that stands in for a buffer. Example: [[papers/shin2017-deep-generative-replay]]. HippoCortex replays at the _hidden-state_ level instead of the input level.
- **Regularisation-based** — penalise changes to weights important for past tasks. Example: EWC ([[papers/kirkpatrick2017-ewc]]).
- **Gradient-projection / architectural** — constrain new-task updates to not interfere with prior-task features. Example: GPM ([[papers/saha2021-gradient-projection-memory]]), [[papers/cheng2024-mamba-cl]], [[papers/lee2025-inf-ssm]].

## Benchmarks used in this project

- Split-CIFAR100 (20 tasks, 5 classes each).
- ImageNet-R.
- Real Go2 Edu task stream (Stage 2): terrain → object interaction → human proximity → mapping.

## Survey

[[papers/delange2022-cl-survey]] is the go-to reference for the taxonomy.
