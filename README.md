# LAIM — Anomaly Detection for Multi-Agent AI Systems

An unsupervised anomaly detector for multi-agent AI systems (MAS), implementing
the **LumiMAS** architecture (Solomon et al., [arXiv:2508.12412](https://arxiv.org/abs/2508.12412);
a copy lives in `docs/lumimas.pdf`). It is the detection component of LAIM
(Live AI Monitoring): it watches AEF execution traces of agent systems and
flags anomalous traces in near-real-time, with per-trace attribution surfaces
for root-cause analysis.

```
AEF trace (span tree: llm / tool / chain / retriever / …, 46-field Parquet contract)
   │
   ├─ s1  data      spans → EPI features (~45–70 operational: durations, tokens,
   │                errors, action entropy…) → semantic embeddings (deepvk/USER-bge-m3,
   │                1024-dim) → anomaly injection (5 LumiMAS classes) → trace
   │                sequences → trace-level split → winsorized robust scaling
   │
   ├─ s2  detector  EPI LSTM-AE (Bi+Uni)  ─┐
   │                SEM LSTM-AE (Bi+Uni)  ─┤→ latents → Combined FMLP-AE
   │                                        → reconstruction error = anomaly score
   │                                        → threshold sweep on VAL (youden/f1/…)
   │                                        → 3-branch calibration → p_anomaly ∈ (0,1)
   │
   ├─ s3  classifier  supervised anomaly-type classifier (stacking ensemble)
   ├─ s4  rca         per-span / per-feature error attribution (RCA seams)
   └─ s5  reports     metrics + HTML reports
```

Training is unsupervised on normal traces only. The decision threshold is
selected on a labeled VAL split (metric configurable); anomalies for VAL/TEST
come from a built-in injector covering the five LumiMAS vulnerability classes:
**Hallucination, Bias, DPI (direct prompt injection), IPI (indirect prompt
injection), MP (memory poisoning)**.

> **Provenance.** This codebase was audited and refactored end-to-end from the
> legacy research code. What was broken, what changed and why, and the honest
> before/after metrics are documented in `FINAL_REPORT.md`; start there if you
> want the history. This README describes how to use the system as it is now.

---

## Repository layout

| Path | What it is |
|---|---|
| `run.py` | **The** entry point — all pipeline commands (see [Usage](#usage)) |
| `configs/` | Run configs (`default.toml`, `smoke.toml`) — TOML, overridable from the CLI |
| `ars/` | The numeric pipeline (JAX/Flax/optax + polars + sentence-transformers) |
| `ars/specification/spec.py` | Single source of truth: 46-field spans schema, sentinels, validation expressions |
| `ars/data/` | Feature definitions, contract validation engine, anomaly injector, synthetic trace generator |
| `ars/stages/` | Pipeline stages s1–s5 |
| `ars/models/` | Detector (LSTM-AE / FMLP-AE, calibration, attribution) and classifier models |
| `laim/` | Orchestration: typed config, logging, run manifests, evaluation surface |
| `tests/` | 75 tests, 12 layers (see [Testing](#testing--verification)) |
| `baseline/` | Frozen legacy baseline + validation-run evidence (JSON) |
| `docs/` | LumiMAS paper + the authoritative data-requirements spec (Russian, v1.9.2) |
| `descriptor.json` | SberDS node descriptor — this repo deploys as one dual-mode (train/inference) platform node via `run.py::main(**params)`; see `deploy/README.md` |
| `deploy/` | Deployment guide (`deploy/README.md`) + archived legacy descriptors |
| `legacy/` | Archived dead code (`CHANGELOG.md` explains each move) |
| `AUDIT_*.md`, `PLAN.md`, `GAPS.md`, `VALIDATION.md`, `TEST_REPORT.md`, `FINAL_REPORT.md` | The audit/refactor record |

---

## Installation

Requirements: **Python ≥ 3.12** (PEP 695 syntax is used throughout), ~6 GB disk
for dependencies. GPU is optional — everything runs on CPU; on CUDA hosts
install the matching torch/jax wheels instead of the CPU ones.

```bash
# with uv (recommended; plain `pip -m venv` works the same way)
uv venv .venv --python 3.12
source .venv/bin/activate

# CPU torch first (skip --index-url on CUDA hosts to get the GPU wheel)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e '.[dev]'
```

Pinned versions that are known-good live in `requirements.txt`
(`uv pip install -r requirements.txt` reproduces the validated environment).

### The embedding model

The semantic branch uses **`deepvk/USER-bge-m3`** (1024-dim, SentenceTransformers
format). On a machine with network access:

```bash
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('deepvk/USER-bge-m3').save('/models/USER-bge-m3')"
```

then point `paths.embedder` at `/models/USER-bge-m3`.

**Offline / air-gapped environments** (like the one this repo was validated in):
build the deterministic random-weight stand-in — same interface, same 1024-dim
unit-norm outputs, no semantics:

```bash
python baseline/build_standin_embedder.py /tmp/models/USER-bge-m3-standin
```

The default configs point at the stand-in path. **Numbers produced with the
stand-in measure pipeline mechanics, not semantic quality** — see
`OPEN_QUESTIONS.md` (OQ-3) and `VALIDATION.md` for exactly what this affects.
Either way, the model directory is fingerprint-pinned into training artifacts;
serving against a different model raises an error rather than silently
producing garbage.

---

## Quick start

```bash
# 1. verify the installation (≈2–4 min, see Testing below for details)
make test

# 2. end-to-end smoke run on the bundled 43-trace sample (≈5–8 min, CPU)
python run.py all --config configs/smoke.toml
```

The smoke run prints its run directory at the end
(`runs/<timestamp>_all_<confighash>`). Inspect it:

```
runs/<id>/
├── manifest.json        # config hash, input SHA-256, seeds, stage timings, metrics
├── run.log              # structured log of the whole run
├── eval_report.json     # full metric surface (see Usage → eval)
├── s1_meta.json         # data-prep metadata incl. embedder fingerprint
├── s2_meta.json         # detector metadata incl. threshold + calibration
└── traces_train/        # split parquets + trained model artifacts + HTML reports
```

---

## Usage

Everything goes through `run.py`. Any config value can be overridden with
repeatable `--set section.key=value` flags; lists use JSON syntax
(`--set 'detector.experiments=["hub_mse_mse_08_4"]'`).

### Generate synthetic traces

```bash
python run.py synth --config configs/default.toml \
    --set synth.target_spans=20000 --set synth.seed=20250601
```

Deterministic generator producing spec-conformant spans (multiple scenarios,
agents, deliberate defects for validation testing).

### Validate a spans file against the data contract

```bash
python run.py validate --spans data/traces_1k_sample.parquet
```

Prints and records per-trace rejection statistics (a trace is rejected when any
span violates a mandatory field of the 46-field contract — see
`docs/data_requirements/`).

### Train

```bash
python run.py train --config configs/default.toml \
    --set paths.train_spans=/data/my_traces.parquet \
    --set detector.epochs=100 --set runtime.device=gpu
```

`prepare` runs only the data stage; `train` = prepare + detector (+ classifier
if `classifier.enabled`); `eval` = train + the full evaluation report;
`all` = eval + inference on `paths.infer_spans` if set.

Key knobs (all in `configs/*.toml`, full list in `laim/config.py`):

| Knob | Default | Meaning |
|---|---|---|
| `runtime.seed` | 12345 | drives every RNG; same seed + hardware ⇒ bit-identical run |
| `runtime.device` | `cpu` | `cpu` / `gpu` |
| `data.validation_gate` | `warn` | `off` / `warn` / `strict` (strict trains only on contract-conformant traces) |
| `data.embedding_max_length` | 1024 | token truncation — the main CPU-speed lever for the real embedder |
| `data.embedding_batch_size` | 32 | encoder batch size (memory vs throughput) |
| `data.inject_anomalies` | `true` | inject the 5 LumiMAS classes for VAL/TEST labeling |
| `detector.experiments` | both grid entries | architecture grid filter (batch size must be ≤ #train traces) |
| `detector.epochs` / `patience` | 10 / 50 | raise for real training (10 is a smoke-scale default) |
| `detector.threshold_metric` | `youden` | metric maximized on VAL for the decision threshold |
| `detector.select_on` | `val` | which split ranks experiments (`test` reproduces the legacy bias — don't) |
| `eval.target_fpr` | 0.10 | recall@FPR≤target reported per run |
| `eval.assumed_prevalence` | 0.01 | production prevalence for PPV projection |

### Evaluate

`run.py eval …` (or any `all` run) writes `eval_report.json` containing, for
VAL and TEST: overall precision/recall/FPR/F1/ROC-AUC/PR-AUC, **per-anomaly-type**
metrics, **recall @ FPR ≤ target** with its threshold, **prevalence-adjusted
PPV**, calibration quality (ECE, Brier, reliability table), and per-trace
latency percentiles (p50/p95/p99, batch=1, after warmup).

### Score new data (inference — separate from training)

```bash
python run.py infer --model-dir runs/<trained_run_id> \
    --spans /data/new_traces.parquet
```

Writes `detections.parquet` with a **full audit trail**: every input trace with
`detector_p_anomaly`, `detector_confidence`, per-branch errors, normalized
latents, `detector_is_anomaly`, `detector_truncated` (traces longer than the
trained sequence budget), and RCA attribution columns
(`rca_top_span_indices/errors`, `rca_top_feature_indices/errors`,
`rca_report_str`). Inference verifies the embedder fingerprint against the
training artifacts and refuses to serve on mismatch.

---

## Testing & verification

Everything below is CPU-only and needs no network. First-ever run on a machine
pays JAX compilation (add ~1–2 min); timings below are for warm caches.

### Step 0 — sanity: the package imports

```bash
.venv/bin/python -c "import ars.main, laim.pipeline; print('imports OK')"
```

Expected: `imports OK` in ≤ 30 s (JAX initialization dominates). If this fails,
your Python is < 3.12 or dependencies didn't install — re-check
[Installation](#installation).

### Step 1 — fast suite (the default gate)

```bash
make test          # == .venv/bin/python -m pytest -q -m "not slow and not gpu" --timeout 240
```

Expected output (≈2–4 min):

```
....................................................................  [100%]
70 passed, 5 deselected in ~120-240s
```

**Any failure here is a real problem** — the suite has no flaky tests. The 5
deselected tests are the `slow` layer (step 2).

What the fast suite covers, per file:

| File | Layer | What a failure means |
|---|---|---|
| `test_data_contract.py` (17) | data contract | schema/sentinel/rejection logic drifted from `docs/data_requirements` |
| `test_characterization_s1.py` / `_s2.py` / `_injector.py` | golden characterization | pipeline numerics changed vs `tests/golden/golden.json` — either you changed behavior intentionally (regenerate, see below) or you broke something |
| `test_anti_leakage.py` (5) | leakage invariants | a train/val/test hygiene property broke — treat as P0 |
| `test_property_based.py` (7) | Hypothesis invariants | metric math, injector totality, normalization round-trip, permutation invariance |
| `test_numerics_and_behavior.py` (most of 12) | numerics + failure modes | NaN guards, sentinel robustness, zero-batch guard, calibration monotonicity |
| `test_config.py` (6) | config layer | TOML/CLI precedence or coercion broke |
| `test_attribution.py` (5) | RCA seams | attribution no longer localizes injected corruption |
| `test_embeddings.py` (2 of 3) | embedder pinning | fingerprint verification broke |

### Step 2 — full suite (adds micro-training, integration, latency, cross-process determinism)

```bash
make test-all      # == pytest -q -m "not gpu"
```

Expected: `75 passed` in ≈8–12 min. The 5 additional tests:

* `test_micro_train_losses_pinned` — 3-epoch training reproduces golden losses bit-for-bit;
* `test_micro_train_bit_identical_across_processes` — same seed ⇒ identical losses in two fresh interpreters (determinism);
* `test_span_embeddings_are_text_dependent` — real embeddings flow per span (builds the stand-in embedder automatically if `/tmp/models/USER-bge-m3-standin` is missing — first run adds ~30 s);
* `test_branch_inference_latency_budget` — per-trace inference p50 < 100 ms CPU;
* `test_full_pipeline_on_fixture` — **the integration test**: raw AEF spans → prepare → train (1 epoch) → eval → infer, asserting manifest stages, eval-report shape, the full audit trail, RCA columns, and p_anomaly ∈ (0,1). ≈4 min.

### Step 3 — end-to-end smoke run (the "does the real thing work" check)

```bash
python run.py all --config configs/smoke.toml
# or: make train-smoke
```

≈5–8 min on 4 CPU cores. Verify the outcome:

```bash
RUN=$(ls -dt runs/* | head -1)
python - <<EOF
import json
m = json.load(open("$RUN/manifest.json"))
print({k: v.get('status') for k, v in m['stages'].items()})
e = json.load(open("$RUN/eval_report.json"))
print('test overall:', e['test']['overall'])
print('latency p50 ms:', e['latency_per_trace']['p50_ms'])
EOF
```

Expected:

* `prepare`, `train_detector`, `eval` → `ok`;
* `train_classifier` → **`failed: ValueError: классификатор: недостаточно примеров на класс …`** —
  this is **correct** on the 43-trace sample: the classifier refuses starved
  classes with an actionable error instead of training garbage (finding F-06).
  It trains fine on realistic corpora (step 4). Set `classifier.enabled=false`
  to silence it;
* eval metrics present and finite; latency p50 well under 100 ms CPU.
* Do **not** read quality metrics off the smoke run — 43 traces and 3 epochs
  measure wiring, not models.

### Step 4 (optional, ≈1 h CPU) — realistic-scale end-to-end

Reproduces the Phase 7 validation (`VALIDATION.md`, evidence JSONs in
`baseline/validation_artifacts/`):

```bash
python run.py synth --config configs/default.toml --set synth.target_spans=20000
# split by trace, train on 90%, keep 10% for inference — see VALIDATION.md §2
python run.py all --config configs/default.toml \
    --set paths.train_spans=/tmp/synth_train.parquet \
    --set paths.infer_spans=/tmp/synth_infer.parquet
```

Expected: all five stages `ok` (incl. the classifier), detector val
recall ≈ 0.40 @ FPR ≈ 0.10 with the stand-in embedder at 10 epochs, per-type
recall ≈ 0.9 for MP / ≈ 0.8 for DPI, IPI/bias near chance (semantics-free
stand-in — expected; see `VALIDATION.md` for the full reading).

### Golden files — the regeneration policy

Characterization tests pin exact numbers at seed 12345 in
`tests/golden/golden.json`. If you **intentionally** change pipeline behavior:

```bash
make golden        # regenerates tests/golden/golden.json
```

then commit the diff **in the same commit as the behavior change, with the
before/after explained in the commit message**. A golden diff you cannot
explain is a bug you introduced. Never regenerate to silence a failure you
don't understand.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| **"Stuck" at `ЭМБЕДДИНГИ: sem_text -> sem_vector`** | It is (almost certainly) working, not stuck — this stage now logs model-load time and per-chunk progress (`эмбеддинги: N/M (…%, X текст/с, осталось ~Y с)`); if you see no progress lines, update to the current branch. Expected CPU rates: stand-in ≈ 100–300 texts/s; **real `USER-bge-m3` ≈ 1–10 texts/s** — on the 1k-span sample that is minutes, on large corpora use a GPU (`runtime.device=gpu`). Levers: only unique texts are encoded (typically 2–50× fewer than spans); `--set data.embedding_max_length=256` cuts real-model CPU time ~4× (truncates long span texts); `--set data.embedding_batch_size=8` reduces memory pressure on small machines. If throughput is ~0 texts/s on the *stand-in*, check for CPU oversubscription (another training run on the same cores) |
| `SyntaxError` on import | Python < 3.12 — the code uses PEP 695 `type` aliases |
| `No module named pip` inside `.venv` | the venv is uv-managed — use `uv pip …` |
| First test run takes 2× longer | cold JAX compile cache — normal; rerun is fast |
| `модель эмбеддера не найдена` | build the stand-in (`python baseline/build_standin_embedder.py /tmp/models/USER-bge-m3-standin`) or point `paths.embedder` at a real model |
| `отпечаток модели эмбеддера не совпадает` | you're serving with a different embedder than the one trained with — intentional guard (F-10); retrain or restore the original model dir |
| `batch_size=32 больше числа обучающих трасс` | corpus too small for that experiment's batch — use the batch-8 experiment (`--set 'detector.experiments=["hub_mse_mse_08_4"]'`) or a bigger corpus (F-40 guard) |
| `классификатор: недостаточно примеров на класс` | expected on tiny corpora — grow the corpus or `--set classifier.enabled=false` (F-06 guard) |
| `FloatingPointError: … невалидная ошибка` | training hit NaN/Inf and aborted deliberately (F-21 guard) — inspect your input data and normalization report |
| Push/CI artifacts too big | `runs/` and `runs_validation/` are gitignored on purpose — model pickles don't belong in git |

---

## Data contract (summary)

The authoritative spec is `docs/data_requirements/` (ТЗ v1.9.2 + reference
validation code). In one paragraph: spans arrive as a 46-field Parquet table;
**no NULLs anywhere** — "not applicable / unknown" is encoded with sentinels
(`-1`, `-1.0`, `false`, `''`, `'root'`, `'outside'`, `'NONE'`,
`'STATUS_CODE_UNSET'`); IDs are base64, `agent_id` matches `C[IE][0-9]+`;
LLM/HTTP/Kafka fields must be sentinel-valued outside their span kinds; one
invalid mandatory field rejects the **whole trace**. `ars/specification/spec.py`
encodes all of it as polars expressions; `run.py validate` applies it;
`data.validation_gate=strict` enforces it before training. Feature engineering
treats sentinels as nulls (skipped by aggregations), per the spec.

## Determinism & reproducibility

Same seed + same hardware ⇒ bit-identical artifacts (verified across
processes; the frozen-baseline comparison scripts live in `baseline/`).
Every run directory carries a `manifest.json` with the config hash, input file
SHA-256s, seeds, package provenance and git SHA — a run is reproducible from
its manifest alone. The embedding model is fingerprinted into artifacts.

## Documentation index

| Doc | Read it for |
|---|---|
| `FINAL_REPORT.md` | the whole story: what was broken, what changed, honest metrics |
| `AUDIT_00_baseline.md` | the frozen legacy baseline and how it was captured |
| `AUDIT_01_findings.md` | every defect (P0–P3) with file:line and proof tests |
| `AUDIT_04_parity.md` | proof the refactor was behavior-preserving before fixes |
| `PLAN.md` / `GAPS.md` | the executed plan and the gap analysis behind it |
| `VALIDATION.md` | baseline-vs-now metrics, per anomaly type, gap analysis vs the 80/10 target |
| `TEST_REPORT.md` | the test matrix and what is honestly untested |
| `OPEN_QUESTIONS.md` / `DECISIONS.md` / `CHANGELOG.md` | decisions, defaults, and the change ledger |
| `CLAUDE.md` | orientation for AI-assisted sessions |

## Known limitations

Validated on CPU with a random-weight stand-in embedder (network-blocked
environment): semantic-class detection (hallucination/IPI/bias) is a lower
bound until re-validated with the real `deepvk/USER-bge-m3` on GPU hardware.
Calibration is monotone but weak (ECE ≈ 0.5 on synthetic data) pending a
dedicated calibration holdout. The s3 classifier's self-reported metrics carry
documented optimism (AUDIT_01, L1–L12). The `deploy/` payloads need
regeneration before platform deployment. Full list with ranking:
`FINAL_REPORT.md` §6.
