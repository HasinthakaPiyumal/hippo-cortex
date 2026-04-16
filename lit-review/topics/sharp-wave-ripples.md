---
tags: [topic, neuroscience, sharp-wave-ripples]
---

# Sharp Wave Ripples (SWRs)

Brief high-frequency bursts in the mammalian **hippocampus**, prominent during quiet wake and slow-wave sleep. They transmit **compressed** memory traces (not raw sensory data) to the neocortex for consolidation.

## Biological picture

During waking hours the hippocampus rapidly encodes new experience. During sleep, SWR bursts fire and carry consolidated traces to the neocortex, which slowly integrates them into stable long-term representations ([[papers/buzsaki2015-swr]]).

Two key properties HippoCortex borrows:

- **No raw storage** — the hippocampus doesn't replay raw sensory data; it replays compressed bursts.
- **Constant memory footprint** — the hippocampus doesn't grow linearly with lifetime experience.

## Mapping to HippoCortex

| Biology                    | HippoCortex analogue                                    |
| -------------------------- | ------------------------------------------------------- |
| Hippocampus fast encoder   | Stats buffer of `(µ, σ²)` of Mamba hidden states.       |
| SWR burst during sleep     | Conditional-VAE SWR generator sampling `Ĥ_past`.        |
| Neocortex long-term store  | Mamba backbone weights (updated via null-space gradient)|

See [[topics/generative-replay]] for the ML framing.
