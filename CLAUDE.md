# CLAUDE.md — orientation for the next session

## What this repo is

LAIM anomaly detection for multi-agent AI systems, implementing the LumiMAS
architecture (paper: `docs/lumimas.pdf`). Unsupervised detector over AEF span
traces: EPI features + semantic embeddings → two LSTM autoencoders → combined
FMLP autoencoder → reconstruction error → calibrated `p_anomaly`. A supervised
anomaly-type classifier (s3) and an RCA seam (s4) sit downstream. Deploy target
is a SberDS-style node platform (`deploy/`): `descriptor.json` points it at
`run.py::main`, which calls `laim.platform.run_node`.

Read in this order: `PLAN.md` **"Active threads"** (current state + agreed
next steps — start here), `AUDIT_00_baseline.md` (what the legacy code did),
`AUDIT_01_findings.md` (every defect, with IDs used across commits/tests),
`AUDIT_05_findings.md` (F-75..F-84 — status table at the top; all fixed except
F-80), `PLAN.md` + `GAPS.md`, `AUDIT_04_parity.md`, `VALIDATION.md`,
`FINAL_REPORT.md`.
For the SberDS deployment (works end-to-end since 2026-08-14): `deploy/README.md`
— every platform quirk with its evidence. Experiments: `EXPERIMENTS.md`.

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
  scores new data (full audit trail + RCA columns: `rca_top_*`, index-space
  `rca_attribution` JSON for every trace, and `detector_rca` for flagged traces).
- `tests/` — ~190 tests; `make test` (fast, CPU, ~3 min warm), `make test-all`
  (adds micro-training/integration/latency). Golden pins live in
  `tests/golden/golden.json`; regenerate ONLY with an intended behavior change
  (`make golden`) and explain the diff in the same commit.
- `baseline/` — the frozen legacy baseline and its tooling.
- `legacy/` — archived dead code (see CHANGELOG.md).
- `rca/` — the LAIM RCA node: a SEPARATE SberDS node (own descriptor.json,
  main.py, requirements, tests; imports nothing from ars/laim). It consumes
  `test_anomalies`, including the `detector_rca` block (schema
  `laim.detector_rca/1`, built by `s4__rca.export_detector_rca`), filters
  false positives with an LLM, and writes the RCA. Tests run in their own env:
  `cd rca && pytest` (see rca/README.md). laim's `tests/test_rca_export.py`
  guards the contract, including that the node's glossary covers every feature.
- `anomaly_report/` — the LAIM anomaly report node: a SEPARATE SberDS node
  (stdlib only) that renders the RCA node's `res` (or `test_anomalies`) into
  the HTML report. It hides anomaly types when s3 did not run (empty
  `anomaly_type`). Tests: `cd anomaly_report && pytest`.

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
- Determinism: everything flows from `runtime.seed` (including the injection
  plan since F-81 and `synth.seed` for the generator); same seed + hardware is
  bit-identical on CPU (verified across processes). GPU runs are NOT verified
  bit-reproducible: no deterministic-XLA flags are set. Never use unseeded RNG.
- The embedding model is fingerprint-pinned (`S1Meta.embedding_fingerprint`);
  serving with a different model directory raises. The real model is
  `deepvk/USER-bge-m3` (1024-dim) and serves on SberDS via the
  `path_embedder` port (confirmed in production). Dev containers use a
  random-weight stand-in because huggingface.co is network-blocked (OQ-3) —
  local quality numbers measure pipeline mechanics, not semantic quality.
- Injection labels are a pure function of trace_id (hash with the plan seed =
  `runtime.seed`) and of `(aef_kind, agent_id)` cell sizes (a class is kept
  only where the trace has spans it can perturb — F-79), computed by ONE
  function (`Assign.trace_labels`) for `planned_trace_labels` and the
  injection. That is how train-only feature selection works pre-injection.
  If you change the injector's label assignment, s1 has a hard runtime
  consistency check that will fail loudly. Per-class coverage (planned /
  no_victims / labeled / applied / unapplied) is in `S1Meta` and
  `eval_report.injection_coverage`; labeled-but-unchanged traces are dropped.
- Config keys are strict: an unknown key in TOML / `--set` /
  `config_overrides` raises with the closest valid names (F-82).
- DEVICE TRAP: `laim.pipeline._apply_runtime` must run before anything
  imports an `ars` stage/data module (they import `c0__env_setup`, which hides
  the GPUs unless `ARS_DEVICE=gpu`; `run_node` pre-sets `ARS_DEVICE` from the
  form so port staging's `ars.specification` import is safe), and no module
  may initialize JAX or probe torch CUDA at import — once the JAX backend
  exists, or cuInit has run, the device set of the process is fixed. Breaking
  this made a `device=gpu` training run silently on CPU (20 min -> ~6 h).
  Heavy GPU-probing imports (`sentence_transformers`) stay inside functions.
  `tests/test_runtime_device.py` guards it (with a flash-attn stub);
  `device=gpu` without a working GPU fails loudly (`verify_runtime_device`).
- COMPILE TRAP: XLA compiles a jitted function once per input SHAPE, and an
  op run outside jit is its own compiled program. On GPU a compile costs tens
  of ms (one op) to seconds (a forward pass): a 127-trace platform run spent
  176 of 220 s compiling. Inference-style calls go through `over_blocks` with
  one row count per training run (`PreparedData.infer_rows`), model init is one
  jitted program (`TRAIN.make_train_state`). Don't call a jitted forward on
  each split's own size; check `metrics.resources.xla_top` (one name with many
  compiles = per-shape recompiles; `add`/`mul`/... = op-by-op code).
  `tests/test_compile_reuse.py` guards it.
- The experiment grid lives in `ars/configuration/experiments/e2__detector.py`;
  `detector.experiments/epochs/patience` in config override it. The grid is
  BUILT from the requested codes (any well-formed code, not only `CODES`;
  malformed codes fail before s1 — F-78), and every experiment uses
  `detector.threshold_metric` (the old per-position metric cycling in
  `build_grid` is gone — F-77). Full catalogue: `EXPERIMENTS.md`.
- `ars/main.py` is the LEGACY platform adapter (pre-`laim` contract) and a
  library: `laim.platform` imports `Anomalies` and `extract_query_response`
  from it. The platform itself calls `run.py::main` -> `laim.platform.run_node`.
  Don't break those imports; `ars/main.py::main` is not the way to run things.
- TUI prints are Russian; logs from `laim` are English. Chart rendering is
  crash-proof (degrades to missing images).
- Metric semantics: the per-trace branch errors (`e_epi`/`e_sem`/`e_comb`,
  what scoring, threshold and calibration use) are the per-element LOSS of
  that branch — Huber for EPI in both default experiments (and for the FMLP
  in `hub_mse_hub_32_4_deep`), MSE otherwise — normalized per element since
  F-23. The `*_mse` diagnostics in `results.json` are plain per-element MSE.
  Old reports' "~1200 MSE" numbers used timestep-normalized loss on unfloored
  robust scales — not comparable.

## Where the bodies were buried (fixed, but instructive)

Zero-stub embeddings made the semantic branch an injection oracle (F-01);
test-set model selection (F-02); std+eps latent normalization overflow (F-03);
anti-calibrated Platt weights (F-04); robust-scale explosion (F-05); u64 hash
wraparound crashing the injector at scale (F-36). Full list: AUDIT_01.

## Remaining known work (see PLAN.md "Active threads" + "Remaining work")

CURRENT FRONTIER: detection quality on the operator's real corpus (first
500-epoch run was near-chance). AUDIT_05 found and fixed the main suspect —
EPI feature selection collapsed to an alphabet prefix (F-75) — plus the
grid ignoring `threshold_metric` (F-77) and dropping non-catalogue codes
(F-78); the next real-corpus run is described in PLAN.md "Active threads".
OQ-8 is done (`data.injection_fractions`, `scale_floor`/`norm_z_clip` wired).
Open: F-80 (report hides flags with p_anomaly < 0.75 — needs a product
decision). Longer-term: out-of-core s1 for 56 GB corpora; experiment-grid
parallelism (no-spawn/no-fork constraint, D-4 addendum); s3 stacking CV
redesign (F-33); drift metrics.
