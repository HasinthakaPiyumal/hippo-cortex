# `checkpoints/`

Trained model weights. **Contents are gitignored** — only `.gitkeep` and this README are tracked.

## Suggested layout

```
checkpoints/
├── stage1_split_cifar100/<run_id>/task_<N>.pt
├── stage1_imagenet_r/<run_id>/task_<N>.pt
├── stage2a_hybrid_vision/<run_id>/...
├── stage2b_sim_mujoco/<run_id>/...
└── stage2c_go2_deployment/<run_id>/...
```

Keep the stats buffer `(µ, σ²)` alongside each task checkpoint — it's cheap and is part of the model state under HippoCortex.

Before releasing the open-source artefact at Month 10, upload the final checkpoints to a release mirror (HuggingFace / Zenodo) and link from the top-level README.
