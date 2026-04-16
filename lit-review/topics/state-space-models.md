---
tags: [topic, state-space-models, architecture]
---

# State Space Models (SSMs)

Neural networks that process sequential data through a **recurrently updated hidden state**, achieving **linear-time** computation in sequence length — a key property for edge robotics.

## Mamba

Mamba ([[papers/gu2023-mamba]]) adds input-dependent (selective) state updates, matching Transformer accuracy at a fraction of the memory cost. It's the backbone for HippoCortex Stage 1 and (paired with Transformer attention) Stage 2.

## SSMs + Continual learning

Two direct baselines for HippoCortex:

- Mamba-CL ([[papers/cheng2024-mamba-cl]]) — null-space optimisation on Mamba.
- Inf-SSM ([[papers/lee2025-inf-ssm]]) — exemplar-free CL for SSMs.

## Why SSMs for the Go2 Edu?

- Linear-time step cost → fits the Jetson Orin NX 16GB / 100 TOPS envelope.
- Recurrent hidden state → good fit for continuous sensor streams (RGB, D435i depth, L1 LiDAR, IMU).
- Hidden state is a natural unit to store statistics `(µ, σ²)` of and regenerate via SWRs.
