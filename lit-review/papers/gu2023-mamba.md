---
title: "Mamba: Linear-time sequence modeling with selective state spaces"
authors: [Albert Gu, Tri Dao]
year: 2023
venue: "arXiv preprint"
arxiv: "2312.00752"
doi:
tags: [state-space-models, mamba, sequence-modeling, architecture]
status: to-read
---

# Summary

Introduces Mamba: an SSM with input-dependent selection that matches Transformer quality at linear compute cost in sequence length.

# Key ideas

- Selective state update (input-dependent A, B, C matrices).
- Hardware-aware parallel scan for efficient training.
-

# Method

# Results

# Relevance to HippoCortex

Proposal reference [5]. Mamba is the backbone for both Stage 1 and (paired with Transformer attention) Stage 2. The stats buffer stores `(µ, σ²)` of _Mamba hidden states_, so we need a solid understanding of the hidden-state geometry.

# Questions & gaps

# Related notes

- [[topics/state-space-models]]
