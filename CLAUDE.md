# CLAUDE.md — orientation for the next session

## What this repo is

LAIM anomaly detection for multi-agent AI systems, implementing the LumiMAS
architecture (paper: `docs/lumimas.pdf`). Unsupervised detector over AEF span
traces: EPI features + semantic embeddings → two LSTM autoencoders → combined
FMLP autoencoder → reconstruction error → calibrated `p_anomaly`. A supervised
anomaly-type classifier (s3) and an RCA seam (s4) sit downstream. Deploy target
is a SberDS-style node platform (`deploy/`), which calls `ars/main.py::main`.

Read in this order: `AUDIT_00_baseline.md` (what the legacy code did),
`AUDIT_01_findings.md` (every defect, with IDs used across commits/tests),
`PLAN.md` + `GAPS.md`, `AUDIT_04_parity.md`, `VALIDATION.md`, `FINAL_REPORT.md`.

## Layout

- `ars/` — the numeric pipeline (JAX/Flax/optax + polars + torch-based
  sentence-transformers). Stages: `s1__data` (load→features→embed→inject→
  split→normalize), `s2__detector` (train/threshold/calibrate/infer),
  `s3__classifier`, `s4__rca` (attribution formatting), `s5__reports`.
  `ars/specification/spec.py` is the single source of truth for the 46-field
  spans schema and sentinels. `ars/data/validation.py` is the contract engine.
- `laim/` — orchestration: typed config (TOML + `--set a.b=c`), logging, run
  manifests, evaluation surface. It wires `ars`, never re-implements numerics.
- `run.py` — the only entry point you need:
  `run.py {synth|validate|prepare|train|eval|infer|all} --config configs/default.toml --set …`
  Training and inference are separate: train produces `runs/<id>/` (models,
  `manifest.json`, `eval_report.json`); `infer --model-dir runs/<id> --spans f.parquet`
  scores new data (full audit trail + RCA columns).
- `tests/` — 94 tests; `make test` (fast, CPU, ~3 min warm), `make test-all`
  (adds micro-training/integration/latency). Golden pins live in
  `tests/golden/golden.json`; regenerate ONLY with an intended behavior change
  (`make golden`) and explain the diff in the same commit.
- `baseline/` — the frozen legacy baseline and its tooling.
- `legacy/` — archived dead code (see CHANGELOG.md).

## How to run

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # or CUDA torch
uv pip install -e '.[dev]'
python baseline/build_standin_embedder.py /tmp/models/USER-bge-m3-standin  # if HF unreachable
make test
python run.py all --config configs/smoke.toml     # minutes, CPU
```

## Data contract

`docs/data_requirements/` (Russian ТЗ v1.9.2 + reference validation code) is
authoritative: 46 fields, sentinels instead of NULL (-1 / -1.0 / False / '' /
'root'/'outside'/'NONE'), whole-trace rejection on any invalid mandatory field.
`run.py validate --spans f.parquet` checks a file; the training gate is
`data.validation_gate = off|warn|strict` (default warn).

## Conventions and known traps

- Python ≥3.12 (PEP 695 `type` aliases). No `pip` module inside the uv venv —
  use `uv pip …`.
- Determinism: everything flows from `runtime.seed`; same seed + hardware is
  bit-identical (verified across processes). Never use unseeded RNG.
- The embedding model is fingerprint-pinned (`S1Meta.embedding_fingerprint`);
  serving with a different model directory raises. The real model is
  `deepvk/USER-bge-m3` (1024-dim); this container used a random-weight
  stand-in because huggingface.co is network-blocked (OQ-3) — quality numbers
  with the stand-in measure pipeline mechanics, not semantic quality.
- Injection labels are a pure hash of trace_id (`planned_trace_labels`), which
  is how train-only feature selection works pre-injection. If you change the
  injector's label assignment, s1 has a hard runtime consistency check that
  will fail loudly.
- The experiment grid lives in `ars/configuration/experiments/e2__detector.py`;
  `detector.experiments/epochs/patience` in config override it. Metric cycling
  in `build_grid` assigns different threshold metrics per experiment index —
  intentional-looking but historically accidental; set explicitly.
- `ars/main.py` is the deploy-platform adapter (legacy contract). Don't break
  its signature; it is not the way to run things locally.
- TUI prints are Russian; logs from `laim` are English. Chart rendering is
  crash-proof (degrades to missing images).
- Metric semantics: branch errors are per-element MSE (comparable across
  EPI/SEM/Combined since F-23). Old reports' "~1200 MSE" numbers used
  timestep-normalized loss on unfloored robust scales — not comparable.

## Where the bodies were buried (fixed, but instructive)

Zero-stub embeddings made the semantic branch an injection oracle (F-01);
test-set model selection (F-02); std+eps latent normalization overflow (F-03);
anti-calibrated Platt weights (F-04); robust-scale explosion (F-05); u64 hash
wraparound crashing the injector at scale (F-36). Full list: AUDIT_01.

## Remaining known work (see PLAN.md "Remaining work")

s3 stacking CV redesign (F-33 documented biases); drift metrics; real-embedder
GPU validation; deploy payload regeneration (`deploy/nodes/_codebase/data/` is
not in the repo; `deploy/deploy.py` is empty).
