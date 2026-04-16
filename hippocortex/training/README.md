# `hippocortex/training/`

Training orchestration — awake phase (learn new task) and SWR / sleep phase (consolidate via generative replay + null-space projection).

## Planned contents

- **Awake loop** — vanilla supervised optimisation on the current task's loader from `../data/`.
- **Sleep / SWR loop** — samples `Ĥ_past` from the stats buffer, projects gradients through `I − UUᵀ`, updates the backbone.
- **Metrics** — per-task accuracy, average accuracy (AA), average forgetting (AF), backward transfer (BWT); lives alongside the trainer or moves to `../utils/metrics.py`.
- **Schedulers / optimisers** — shared utilities used by both Stage 1 benchmarks and Stage 2 deployment.

## Invariants to preserve

- Constant memory footprint: never append raw samples to a growing buffer.
- Linear-time per step: compatible with on-board inference on the Jetson Orin NX.
