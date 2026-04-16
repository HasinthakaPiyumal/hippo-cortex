# `notebooks/`

Exploratory Jupyter notebooks — quick experiments, sanity checks, figure generation for papers.

## Conventions

- Name notebooks `YYYY-MM-DD_slug.ipynb` so they sort chronologically and it's clear who wrote what.
- Once a notebook produces something reusable, migrate the logic into `hippocortex/` and keep the notebook as a frozen record.
- Don't commit large cell outputs — clear outputs before pushing (`jupyter nbconvert --clear-output`), or enable `nbstripout`.
