---
title: "Gradient projection memory for continual learning"
authors: [Gobinda Saha, Isha Garg, Kaushik Roy]
year: 2021
venue: "ICLR 2021"
arxiv: "2103.09762"
doi:
tags: [continual-learning, gradient-projection, null-space, regularisation]
status: to-read
---

# Summary

Constrains new-task gradients to lie in the null space of the feature subspace used by prior tasks (`P = I − UUᵀ`), preventing interference without storing raw samples.

# Key ideas

- Maintain orthonormal basis `U` of the span of prior-task activations.
- Project gradients: `∇⊥ = (I − UUᵀ)∇`.
- Update `U` after each task with SVD on activation subspace.

# Method

# Results

# Relevance to HippoCortex

Proposal reference [7]. HippoCortex uses GPM's projector directly — it's the "Null-Space Projector" block in Figure 2 of the proposal. Understand the exact criterion they use to grow `U` across tasks; we may need to adapt it for Mamba hidden states rather than CNN activations.

# Questions & gaps

# Related notes

- [[topics/null-space-projection]]
- [[topics/continual-learning]]
