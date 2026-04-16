---
title: "Learning structured output representation using deep conditional generative models"
authors: [Kihyuk Sohn, Honglak Lee, Xinchen Yan]
year: 2015
venue: "NeurIPS 2015"
arxiv:
doi:
tags: [generative-models, vae, conditional-vae]
status: to-read
---

# Summary

Conditional Variational Autoencoder: VAE whose encoder and decoder are conditioned on an auxiliary variable `y`, enabling structured / class-conditional sample generation.

# Key ideas

- Conditional prior `p(z|y)` and decoder `p(x|z, y)`.
- Training via conditional ELBO.
-

# Method

# Results

# Relevance to HippoCortex

Proposal reference [6]. The SWR generator in HippoCortex is a cVAE conditioned on the task identifier, sampling `Ĥ_past` from the per-task `(µ, σ²)` buffer.

# Questions & gaps

# Related notes

- [[topics/generative-replay]]
