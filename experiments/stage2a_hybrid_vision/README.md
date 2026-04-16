# stage2a_hybrid_vision

Hybrid Mamba + Transformer architecture — Mamba for temporal state across sensor frames, selective Transformer attention for rich spatial features from RGB + D435i depth. Trained on vision datasets before any robot comes into play.

## Goal

Confirm that the HippoCortex SWR + null-space projector wraps the hybrid stack cleanly (Figure 3 in the proposal) without architectural modification.

## Outputs

Runs write to `../../results/stage2a_hybrid_vision/<run_id>/`.
