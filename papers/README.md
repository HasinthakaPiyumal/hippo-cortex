# `papers/`

LaTeX manuscripts the team writes. Two planned:

- `stage1-swr-replay/` — Stage 1 paper: HippoCortex SWR generative replay on Mamba SSM. Submission target TBD (candidate venues: NeurIPS / ICLR / ICML workshops, TMLR).
- `stage2-embodied-cl/` — Stage 2 paper: embodied continual learning on the Go2 Edu. Submission target TBD (candidate venues: ICRA, RSS, CoRL).

## Conventions

- Each paper lives in its own folder with `main.tex`, `references.bib`, `figures/`, and a paper-local `README.md` once started.
- Pull final figures from `../results/<experiment>/<run_id>/plots/` into `papers/<paper>/figures/` rather than linking across folders.
- Keep BibTeX keys consistent with `lit-review/papers/` filenames: `gu2023-mamba`, `saha2021-gpm`, etc.
