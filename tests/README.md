# `tests/`

Pytest unit tests for `hippocortex/`.

## Conventions

- Mirror the package layout: `tests/models/`, `tests/cl/`, etc.
- Test file names: `test_<module>.py`; test function names: `test_<behaviour>`.
- Favour small, deterministic fixtures — no real dataset downloads in unit tests; mock the `../data/` loaders.
- Heavy integration / GPU / robot tests belong in `experiments/`, not here.
