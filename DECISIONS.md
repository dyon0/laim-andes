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

## D-3: SberDS packaging — never ship torch; pin JAX to the CUDA-12 plugin

Grounded in the environment probe of 2026-08-13 (`deploy/probe_node/`, full log
analyzed the same day):

- The `py312-gpu` image PREINSTALLS a working `torch==2.8.0+cu128` (plus
  flash-attn and the complete `nvidia-*-cu12` wheel set). The first deployment
  crash (`ImportError: libsycl.so.9`) happened because our then-`requirements.txt`
  pinned bare `torch==2.13.0`: the sber mirror carries `+xpu` (Intel SYCL)
  builds next to `+cu128`, and PEP 440 ranks `+xpu` above `+cu128` for the same
  version, so pip replaced the working torch with an Intel build whose oneAPI
  runtime the image lacks. **Decision: requirements.txt must never mention
  torch.** The image provides it; sentence-transformers' `torch>=…` floor is
  satisfied by the preinstalled build, so pip does not touch it.
- The hosts run driver 570.86.15 = CUDA 12.8 ceiling. CUDA-13 wheels can never
  initialize there. **Decision: the JAX GPU stack is `jax==0.11.0` +
  `jax-cuda12-plugin[with-cuda]==0.11.0` + `jax-cuda12-pjrt==0.11.0`** (the
  cu12 plugin exists for our exact jax version, so no downgrade); the previous
  cu13 pins (`nvidia-*-cu13`, `cuda-toolkit==13.*`, `triton`) are removed.
- `numpy/pandas/pyarrow/tqdm/psutil` are pinned to the IMAGE's versions
  (2.4.2 / 2.3.3 / 21.0.0 / 4.67.3 / 7.2.2) so the platform's own pywrapper —
  which runs in the same interpreter and itself uses pandas — is never upgraded
  underneath.
- Verification: a clean venv with `torch==2.8.0` + the new requirements
  resolves with torch untouched, and the full fast test suite (87 tests)
  passes against that exact stack (numpy 2.4.2, pandas 2.3.3, pyarrow 21,
  transformers 5.15, jax 0.11). GPU initialization itself can only be verified
  on the platform (no GPU in this container).
- `requirements.txt` is now the PLATFORM install manifest; local development
  uses `uv pip install -e '.[dev]'` per README.
