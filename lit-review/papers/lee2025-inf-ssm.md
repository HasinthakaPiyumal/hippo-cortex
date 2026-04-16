---
title: "Exemplar-free continual learning for state space models"
authors: [I. N. Lee, L. Mahmoodi, T. Le, M. Harandi]
year: 2025
venue: "arXiv preprint"
arxiv: "2505.18604"
doi:
tags: [continual-learning, state-space-models, exemplar-free]
status: to-read
---

# Summary

Exemplar-free (no raw-sample buffer) continual learning for SSMs — the direct spiritual cousin of HippoCortex, with a different mechanism.

# Key ideas

-
-
-

# Method

# Results

# Relevance to HippoCortex

Proposal reference [9]. **Secondary Stage-1 baseline**; target is ~4% average accuracy gain over Inf-SSM. Aligns with HippoCortex on the "no raw-sample buffer" constraint — we need to be explicit about what our SWR generator + stats buffer adds beyond their approach.

# Questions & gaps

- What do they use in place of a buffer? How does their mechanism compare in constant-memory terms?

# Related notes

- [[papers/cheng2024-mamba-cl]]
- [[papers/gu2023-mamba]]
- [[topics/state-space-models]]
