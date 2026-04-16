---
tags: [topic, continual-learning, null-space]
---

# Null-space gradient projection

Constrain new-task gradients so they live in the **null space** of the feature subspace used by prior tasks. Formally:

`∇⊥ = P ∇,   P = I − U Uᵀ`

where `U` is an orthonormal basis of the span of prior-task activations. Gradients in directions already used by prior tasks are zeroed out; new learning is forced into unused directions.

## Key paper

[[papers/saha2021-gradient-projection-memory]] (GPM) — proposed the method on CNN activations. Maintain `U` by SVD on activation matrices collected after each task.

## Applied to SSMs

- [[papers/cheng2024-mamba-cl]] — null-space optimisation on Mamba hidden states (primary HippoCortex baseline).
- [[papers/lee2025-inf-ssm]] — different mechanism, same exemplar-free goal.

## Role in HippoCortex

The "Null-Space Projector" block in Figure 2 of the proposal. It receives `Ĥ_past` from the SWR generator and deflects the new-task update so the Mamba backbone keeps old representations intact while learning the new task.

Paired with [[topics/generative-replay]] — the projector alone needs task-representative features; the SWR generator provides them without a raw buffer.
