---
tags: [topic, continual-learning, generative-replay]
---

# Generative replay

Train a generator alongside the classifier so that "past experience" can be replayed from **synthetic samples** instead of a raw-sample buffer. Sidesteps the memory and privacy costs of buffering.

## Input-level vs. hidden-state-level

| Level         | Who                                        | Pros                          | Cons                                    |
| ------------- | ------------------------------------------ | ----------------------------- | --------------------------------------- |
| Input-level   | [[papers/shin2017-deep-generative-replay]] | Agnostic to downstream model. | Hard to generate realistic high-dim inputs (images, LiDAR). |
| Hidden-state  | **HippoCortex**                            | Much lower-dim target; SSM hidden states are compact. | Generator must track the model's evolving representation space. |

## HippoCortex's SWR generator

A **conditional VAE** ([[papers/sohn2015-cvae]]) conditioned on the task identifier, trained to sample from the per-task `(µ, σ²)` statistics of Mamba hidden states. Outputs `Ĥ_past`, which the null-space projector ([[topics/null-space-projection]]) then uses to deflect new-task gradients.

## Related

- [[topics/continual-learning]]
- [[topics/sharp-wave-ripples]]
