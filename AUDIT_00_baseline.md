# AUDIT 00 — Orientation and Baseline Freeze (Phase 0)

Date: 2026-08-12 · Auditor: automated (Claude) · Branch: `claude/lumimas-anomaly-refactor-7stpda`
Baseline commit of legacy code: `e99d74e` (== branch `legacy`).

---

## 1. What this repository is

An implementation of the **LumiMAS** anomaly-detection architecture (Solomon et al.,
arXiv:2508.12412) as a component of LAIM (Live AI Monitoring), targeting the SberDS
node-graph ML platform. Pipeline:

```
AEF spans (Parquet, 46-field contract, sentinels instead of NULL)
  └─ s1  data: load → EPI features (polars exprs) → [zero-stub] semantic embeddings
          → anomaly injection (5 LumiMAS classes) → trace assembly → split → robust scaling
  └─ s2  detector: EPI LSTM-AE + SEM LSTM-AE → latents → Combined FMLP-AE
          → threshold sweep on VAL → 3-branch calibrated p_anomaly (robust z + Platt)
  └─ s3  classifier: anomaly-type classification (stacking over branch features)
  └─ s4  RCA: **stub** (returns '-dummy-')
  └─ s5  reports: collects holdout metrics + HTML reports
```

Entry point: `ars/main.py::main(path_traces_train, path_traces_test, path_embedder, **ui)`
(the deploy descriptor `env-end2end.json` calls it as `run_all.py::main` with `s1_*`/`s2_*`/`s3_*`
prefixed UI parameters).

## 2. Repo map

| Area | Files | State |
|---|---|---|
| `ars/specification/` | `spec.py`, `core.py`, `common_core.py` | Single source of truth for the 46-attr spans schema, sentinels, validation exprs, recast. **Live.** |
| `ars/data/` | `features.py` (feature defs), `validation.py` (quality/rejection engine), `anomalies_injection.py` (copula/manifold injector, 5 classes), `synthesis.py` (deterministic trace generator), `stages_meta.py` | All live except: **`validation.py` is never called by the pipeline** (only by `synthesis.__main__` diagnostics and its own deploy node). |
| `ars/stages/` | `s1__data.py` (1182 ln), `s2__detector.py` (845 ln), `s3__classifier.py`, `s4__rca.py` (stub), `s5__reports.py` | Live. God-modules; heavy TUI/log entanglement. |
| `ars/models/` | `m2__detector/` (architecture, train, confidence), `m3__classifier/`, `metrics.py` | Live. JAX/Flax. |
| `ars/configuration/` | `c0__*` (env/device), `c1__data.py`, `c2__detector.py`, `c3__classifier.py`, `experiments/` | Live. Frozen dataclasses; overrides via string-prefix kwargs from `main`. `c0__paths.txt`, `c4__rca.py`, `c5__reports.py` are **empty files**. |
| `ars/tools/` | composition DSL (recursion schemes), tui (3 files, color schemes), viz (altair), perf, sql query builder, utilities | Mostly live via imports; composition DSL used only to spell `detect >> classify >> analyze` (main.py:220). |
| `deploy/` | 10 descriptors (5 pipelines × src/env), worker/codebase nodes | SberDS-family node-graph platform. `deploy.py` is **0 bytes**; the packed per-worker source trees (`deploy/nodes/_codebase/data/…`, incl. the `run_all.py` the platform actually executes) are **not in the repo**. |
| `docs/` | `lumimas.pdf` (the paper), `data_requirements/` (Russian data spec v1.9.2 + reference validation code) | Authoritative data contract. |
| `data/` | `traces_1k_sample.parquet` | 1000 spans = **43 traces**, 1 agent, 3 sessions. Schema matches spec field-for-field; dtype drift: `meta_langgraph_step` Float64 (spec: Int64), Categorical vs Enum. |

Dead / vestigial: commented-out function bodies kept inline in `s1__data.py` (3 large blocks),
Russian-named zero-byte TODO files (`ars/models/todo__…`, `ars/specification/todo__…`),
`ars/tools/utilities/env_install.py` (offline wheel installer, referenced nowhere).

## 3. What it took to make it run (environment reconstruction)

Recorded fixes — the repo had **no dependency manifest of any kind**:

1. **Python ≥3.12 required** (PEP 695 `type` statements throughout; deploy images are
   `py312-gpu`). Container default was 3.11 → venv on 3.12.
2. Installed from imports: `polars jax[cpu] flax optax torch sentence-transformers
   transformers scikit-learn altair vl-convert-python psutil pyarrow tqdm pandas`.
   Exact resolved versions: `baseline/runs/*/pip_freeze.txt`.
3. **huggingface.co blocked** by network policy → `deepvk/USER-bge-m3` unobtainable.
   Built a seeded random-weight stand-in (`baseline/build_standin_embedder.py`) with the
   same interface. Code path unchanged (see OQ-3). Note the legacy pipeline stubs all
   normal-span embeddings to zeros anyway (see finding B-1 below).
4. No GPU in this container (brief says 2×H100 target). All numbers below are CPU.
   Device selection code (`ars/configuration/c0__device.py`) honors `device=''` → cpu.
5. There is **no test suite anywhere** (`0 test files`) — nothing to get green.

## 4. Baseline runs

Two corpora, same unmodified pipeline, fixed seed 12345 everywhere:

* **real** — `data/traces_1k_sample.parquet`, split 80/20 at trace boundaries
  (34 train-file traces / 9 test-file traces). Numbers are honest but high-variance:
  after injection+split the model trains on **18 normal traces**.
* **synth** — corpus from the repo's own deterministic synthesizer
  (`GenConfig(target_spans=20000, seed=20250601)`), same split protocol.

Committed experiment grid (`e2__detector.py`): 2 experiments
(`hub_mse_mse_08_4`, `hub_mse_hub_32_4_deep`), `epochs=10` (cut from 500 in-code, OQ-5),
threshold metric = youden for exp 1 / **precision** for exp 2 (an artifact of
`build_grid(CODES, metrics=('youden','precision','recall','f1'))` cycling by index).

Full metrics: **`baseline/baseline_metrics.json`** (real) and
**`baseline/baseline_metrics_synth.json`** (synthetic). Headlines below.

### 4.1 Headline: the pipeline does NOT run end to end

**Finding #1 (the mission's "if it cannot be run, that is finding #1"):** on the real
sample the run crashes in stage s3 (`prepare_features` → `Pad.split` on an empty
dataframe → `ValueError: Need at least one array to concatenate`), because the
classifier re-splits the anomaly pool 60/20/20 stratified with no minimum-count guard
(`s3__classifier.py:144-151,164`) and small classes produce empty splits, and because
`Pad.split` cannot handle 0 rows (`s2__detector.py:63-78`). Stages s1–s2 complete and
write all detector artifacts; the run dies before `run_result.json`.
End-to-end therefore **fails on the shipped data sample**; detector-only baseline
numbers below are recovered from the on-disk artifacts (read-only,
`baseline/eval_baseline.py`).

### 4.2 Real corpus (43 traces), detector baseline — seed 12345, CPU

Best experiment (picked by the pipeline itself, on TEST — see B-2): `hub_mse_mse_08_4`.
Selected threshold: 0.2831 (val sweep, youden).

| Split | n | EPI recon MSE | SEM recon MSE | Combined recon |
|---|---|---|---|---|
| train | 18 | 20.85 | **0.0000** | 0.147 |
| val | 6 | 21.22 | **0.0000** | 0.269 |
| test | 10 | 20.21 | 0.0095 | **392,328,128** |

* TEST overall @ selected threshold: precision 0.80, recall 0.80, FPR 0.20, F1 0.80,
  ROC-AUC 0.80, PR-AUC 0.835 (5 anomalous / 5 normal traces — high variance).
* Per anomaly type (1 trace each — directional only): hallucination ROC-AUC 1.0,
  mp 1.0, bias 0.8, ipi 0.8, dpi 0.4.
* Calibration of `p_anomaly`: **ECE 0.80, Brier 0.80** — anti-calibrated
  (see §4.4; the sign of `w_comb` is negative).
* Latency per trace (batch=1, CPU, warmed): p50 40.6 ms, p95 43.1 ms, p99 92.2 ms.
  (Target: 20–30 ms on the reference GPU; CPU number recorded for trend only.)
* Wall clock: s1+s2 ≈ 5 min on 4 CPU cores (exp1 65 s, exp2 'deep' ≈ 3.5 min).

Full detail: `baseline/baseline_metrics.json`.

### 4.3 Synthetic corpus baseline

<!-- SYNTH_NUMBERS -->

### 4.4 Diagnosed numeric pathologies behind the baseline numbers

These were traced to root cause during Phase 0 because they make the raw numbers
otherwise unreadable; full findings in AUDIT_01.

1. **Robust-scale explosion.** Winsorized robust scaling guards only
   `|IQR| < eps_normalization = 1e-6` (`s1__data.py:836-839`). On the real corpus the
   smallest surviving scale is `avg_word_length_sem_max = 0.0044`, producing
   normalized feature values up to **|42|** (0.5% of all values exceed |10|).
   Squared, this explains EPI MSE ≈ 20 here and the historically reported ~1200 MSE
   on larger corpora: reconstruction error is dominated by a handful of
   quantile-degenerate features, not by model quality.
2. **Latent-normalization divide-by-epsilon.** All train semantic inputs are zeros
   (B-1) → identical sem latents → `sem_latent_std = 1e-8` (exactly the eps) for
   every dimension (`s2__detector.py:306-310`). At inference any nonzero sem latent
   is divided by 1e-8 → combined-branch input z ~ 1e8 → the 392M test error.
   There is no NaN/Inf/overflow guard anywhere downstream.
3. **Calibration inversion.** `sem_mad = 0` → `z_sem = e_sem × 1e8`; BFGS Platt fit
   on 6 val points with such features yields `w_epi=+217, w_sem=-89, w_comb=-209,
   bias=+44.5` — i.e. **higher combined reconstruction error lowers p_anomaly**.
   ECE/Brier of 0.80 are the arithmetic consequence.

## 5. Findings already established during Phase 0

These are expanded with severities and proof plans in `AUDIT_01_findings.md`; recorded
here because they shape how the baseline numbers must be read.

* **B-1 (P0): semantic embeddings are stubbed to constant zeros.**
  `ars/stages/s1__data.py:504-505` — `compute_semantic_embeddings` returns
  `pl.lit((0.0,)*1024)` for every span; the real embedding code below the `return` is
  unreachable. Meanwhile the anomaly injector receives a **real** embedder
  (`s1__data.py:1033`) and (a) replaces hallucination-span vectors with real non-zero
  embeddings (`anomalies_injection.py:712-728`), (b) fits its semantic manifold on the
  zero vectors, so its scale is 0 and **ipi/bias/mp/dpi semantic shifts are no-ops**
  (`anomalies_injection.py:590-594` — `sem + (tau*0)*off`). Net effect: the SEM branch
  is a *was-it-injected-by-the-hallucination-operator* oracle, not a semantic detector.
* **B-2 (P0): model selection reads TEST.** `s2__detector.py:535-538,550` picks the best
  experiment by `test_metrics[select_metric]`.
* **B-3 (P0): IPI and Bias injections are undetectable by construction** (given B-1):
  `touch_epi=False` for both (`anomalies_injection.py:174-178`), semantic shift is a
  no-op ⇒ their injected traces are byte-identical to normals except the label.
  Per-type baseline recall confirms (see metrics JSON).
* **B-4 (P1): the data-contract validation engine is never invoked** in the train/infer
  path — `load_spans` (`s1__data.py:101-211`) reads any parquet blind; `ars/data/validation.py`
  implements the spec's per-trace rejection but only its own deploy node calls it.
* **B-5 (P1): split is not grouped purely by trace_id.** Traces are rows of
  `group_by('trace_id','agent_id')` (`s1__data.py:602`); a multi-agent trace becomes
  several rows that can land in different splits (leakage via shared trace-level
  features). Single-agent in both baseline corpora, so it does not bite here.
* **B-6 (P1): feature selection (fill/static/corr + `max_abs_threshold=1e6`) is fit on
  the full pre-split dataset** (`s1__data.py:398,414`), i.e. selection sees future
  VAL/TEST and injected anomalies.
* **B-7 (P2): `duration_diff` is identically zero** — `features.py:107-111` computes
  `(end-start) - (end-start)`.
* **B-8 (P2): run directory `/tmp/mas-monitor#<second>` with `exist_ok=False` and a bare
  `except: exit(1)`** (`ars/main.py:154-162`) — second run in the same second dies with
  no cause printed; the bare except swallows everything else too.

## 6. Determinism check

Two identical-seed runs of the real-corpus baseline were compared on the
model-relevant artifacts (thresholds, test metrics, reconstruction errors):

**Result: identical.** Two same-seed runs (`baseline/runs/real_a`, `real_b`) produced
bit-equal per-epoch loss curves, thresholds, val/test metrics for both experiments
(`baseline/check_determinism.py` → `"identical": true`). Determinism therefore holds
for the s1→s2 path on this hardware (CPU, single process).

Known residual hazards that this single check does NOT cover (audit items):
run artifacts embed unseeded `run_id` (`ars/main.py:157`) and wall-clock paths;
polars `group_by` iteration order feeds `pl.concat` in the anomaly split
(`s1__data.py:745-748`) — deterministic today via keyed hashing, but unspecified by
polars' contract; multi-GPU/GPU nondeterminism is untested (no GPU available).

## 7. Interpretation — what the baseline numbers mean

The metric that the legacy pipeline itself reports as "best" is not comparable to the
production targets (Recall ≥ 80% @ FPR ≤ 10%) because:

1. VAL/TEST anomaly prevalence is the injector's ~20%, not production prevalence;
2. per-type recall is dominated by trivially-detectable hallucinations (B-1) and
   undetectable ipi/bias (B-3);
3. the experiment picked "best" was chosen on TEST (B-2);
4. with 10 epochs on tens of traces (real corpus), the AEs are barely trained.

The baseline is therefore frozen primarily as a **behavioral reference for the
refactor** (characterization target), not as a quality claim.
