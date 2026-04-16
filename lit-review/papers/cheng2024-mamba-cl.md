---
title: "Mamba-CL: Optimizing selective state space model in null space for continual learning"
authors: [De Cheng, et al.]
year: 2024
venue: "arXiv preprint"
arxiv: "2411.15469"
doi:
tags: [continual-learning, mamba, state-space-models, null-space]
status: to-read
---

# Summary

Applies null-space gradient projection to a Mamba backbone for continual learning — the most direct baseline for HippoCortex's Stage-1 claims.

# Key ideas

-
-
-

# Method

# Results

# Relevance to HippoCortex

Proposal reference [8]. **Primary Stage-1 baseline**; target is ~8–10% average accuracy gain over this on Split-CIFAR100. Read this very carefully — understanding what Mamba-CL _doesn't_ do (no generative replay of hidden states) is how we justify the SWR generator.

# Questions & gaps

- How does Mamba-CL compute `U` on Mamba hidden states vs. CNN activations (cf. Saha et al. 2021)?
- Can their code be reused directly as a baseline in `experiments/stage1_split_cifar100/`?

# Related notes

- [[papers/saha2021-gradient-projection-memory]]
- [[papers/gu2023-mamba]]
- [[topics/state-space-models]]
- [[topics/null-space-projection]]
