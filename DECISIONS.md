# DECISIONS

Engineering decisions with rationale. New heavy dependencies are justified here.

---

## D-1: Environment reconstruction (Phase 0)

The repo ships **no dependency manifest** (no requirements.txt / pyproject.toml /
lockfile; only an offline wheel-install helper in `ars/tools/utilities/env_install.py`).
The environment was reconstructed from imports:

- Python 3.12 (`type` alias statements — PEP 695 — require ≥3.12; deploy descriptors
  use `py312-gpu` base image, confirming 3.12)
- polars, jax[cpu], flax, optax, torch (CPU use), sentence-transformers, transformers,
  scikit-learn, altair + vl-convert-python, psutil, pyarrow, tqdm, pandas

Resolved versions are frozen in `baseline/runs/*/pip_freeze.txt`. No dependency was
added beyond what the legacy code already imports (vl-convert-python is altair's PNG
export backend, required by `ars/tools/visualisations/viz.py`).

## D-2: Baseline instrumentation lives in `baseline/`, not `ars/`

Per-anomaly-type metrics, calibration quality (ECE/Brier) and latency percentiles are
not produced by the legacy pipeline (that itself is an audit finding). The baseline
captures them with read-only post-hoc scripts under `baseline/` that load the saved
artifacts and call existing `ars` functions. No `ars/` source is modified before the
Phase 3 safety net exists.
