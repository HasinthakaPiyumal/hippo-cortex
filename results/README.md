# `results/`

Experiment outputs: per-run metric logs, plots, tables, tensorboard event files. **Contents are gitignored** — only `.gitkeep` and this README are tracked.

## Suggested layout

```
results/
├── <experiment_folder>/
│   └── <run_id>/
│       ├── metrics.json        # AA, AF, BWT per task
│       ├── stdout.log
│       ├── config.yaml         # copy of the config used (for reproducibility)
│       ├── plots/
│       └── tb/                 # tensorboard event files
```

Promote figures from here to `../papers/<stage>/figures/` when they make it into a manuscript.
