# VALIDATION (Phase 7)

Full clean-state runs of the refactored pipeline (`run.py`), CPU container
(4 cores, no GPU — OQ-2), stand-in embedder (OQ-3), seed 12345.
Runs live under `runs_validation/` with complete manifests.

Two corpora:

* **REAL** — the shipped 43-trace sample, same train file as the frozen
  baseline (`baseline/runs/real_a/train_spans.parquet`). Statistically weak by
  construction (18 train-normal traces); kept for side-by-side honesty.
* **SYNTH** — 20 000-span corpus from the repo's own generator (≈950 traces,
  seed 20250601), 90/10 train/infer split by trace. This corpus was
  **impossible to produce before the F-36 fix** (the legacy baseline could not
  run it at all).

## 0. A validation run immediately caught a latent P0 (F-40)

The first REAL validation run aborted at epoch 1 with the new finiteness guard:
the `hub_mse_hub_32_4_deep` experiment (batch 32) has **zero full batches** on
18 train traces — the legacy pipeline trained NOTHING there (loss curves are
literally `[nan, nan, …]` in the frozen baseline artifacts) and silently
evaluated + ranked the random init (baseline "test youden 0.40"). Training now
refuses with a typed error; the REAL validation below uses the batch-8
experiment, the only one that can legitimately train on 18 traces. This is the
strongest single piece of evidence that the guards work.

## 1. REAL corpus — side-by-side with the frozen baseline

Same input file, same seed. "Baseline" = detector-only numbers recovered from
legacy artifacts (AUDIT_00 §4.2); the legacy end-to-end run itself crashed in
s3. Both runs: 10 epochs — quality numbers on 10 test traces carry huge error
bars; the point of this table is *sanity*, not quality claims.

| | Legacy baseline | Refactored | Why it moved |
|---|---|---|---|
| End-to-end run completes | **No** (s3 crash) | **Yes** (classifier stage refuses with actionable error, run completes) | F-06 |
| Combined recon error, test mean | **392,328,128** (overflow) | 0.126 (finite, per-element) | F-03 + F-23 |
| SEM branch error on train/val | 0.0 exactly (zero-stub oracle) | > 0, text-dependent | F-01 |
| Calibration `w_comb` | **−209** (anti-monotone) | +0.6 (priors fallback: val too small for a fit — honest) | F-04 |
| p_anomaly ECE / Brier (test) | 0.80 / 0.80 | 0.25 / 0.24 | F-04 (still poor — 6-point val cannot calibrate; see §3) |
| Test recall / FPR @ selected threshold | 0.80 / 0.20 | 1.00 / 0.80 | threshold now honest-but-noisy: picked by youden on a 6-trace VAL |
| Test ROC-AUC | 0.80 | 0.64 | de-leaked semantics (F-01) — the baseline number was partly the oracle |
| Recall @ FPR ≤ 0.10 (test) | not produced | 0.20 | new metric surface (M9) |
| PPV at 1% prevalence @ threshold | not produced | 0.012 | quantifies F-22: a youden threshold from ~50% VAL prevalence does not transfer |
| Per-trace latency p50 (CPU) | 40.6 ms | 83.8 ms* | *measured while the SYNTH run was training on the same 4 CPUs; idle-machine smoke run: 42.2 ms — treat 42 ms as the CPU reference |
| Deep experiment | "youden 0.40" from an untrained net | refuses to run on 18 traces (typed error) | F-40 |

The honest summary for REAL: after removing the leakage and the numeric
pathologies, a 10-epoch model on 18 traces detects little — as it should.
The legacy numbers looked better *because* they were broken.

## 2. SYNTH corpus (the first realistic-size run this codebase has ever completed)

Run `runs_validation/20260812_125834_all_43d0159ce118` — all five stages
completed: prepare 411 s · detector 1937 s · classifier 921 s · eval 14 s ·
infer 15 s (≈55 min wall on 4 CPU cores). 20 000 spans ≈ 950 traces; detector
trained on 770-trace split (≈440 train-normal traces); both experiments ran
(batch 32 is valid here); selection on VAL chose `hub_mse_mse_08_4`
(threshold 0.0333, youden).

**Detector, VAL (n=626, 50.2% anomalous — injector prevalence):**
recall 0.40 · FPR 0.099 · precision 0.80 · ROC-AUC 0.65 · PR-AUC 0.66.
Recall @ FPR ≤ 0.10 = **0.40** (the youden point happens to sit at the target FPR).

**Detector, TEST (n=452, 30.5% anomalous):**
recall 0.40 · FPR 0.121 · precision 0.59 · F1 0.48 · ROC-AUC 0.61 · PR-AUC 0.45.
Recall @ FPR ≤ 0.10 = 0.10 (the val-chosen operating point degrades on test —
honest small-corpus variance, reported rather than hidden).

**Per anomaly type (test; val in parentheses):**

| Type | n test | Recall | ROC-AUC | Reading |
|---|---|---|---|---|
| mp (memory poisoning) | 22 | **0.91** (0.96) | 0.87 (0.90) | EPI drift — strongest class |
| dpi | 26 | **0.81** (0.71) | 0.81 (0.80) | EPI marginal break — detects well |
| hallucination | 40 | 0.15 (0.27) | 0.51 (0.60) | real text-corruption re-embedding gives modest signal only — the stand-in embedder carries no semantics |
| ipi | 24 | 0.21 (0.15) | 0.44 (0.52) | semantic-only injection ≈ chance with a semantics-free embedder |
| bias | 26 | 0.12 (0.07) | 0.46 (0.48) | same as ipi |

This is the *fixed* behavior: before F-01 the per-type numbers were an
artifact (hallucination AUC 1.0 as an injection oracle, ipi/bias no-ops).
Now the detector detects exactly what its inputs can carry: operational (EPI)
anomalies. Semantic classes await the real `deepvk/USER-bge-m3` (OQ-3).

**Calibration:** ECE 0.47 (val) / 0.65 (test), Brier 0.47/0.64. The guarded
Platt fit is monotone (w_comb=+0.097 > 0) but poorly calibrated — measured and
reported. Follow-up recorded in PLAN remaining work: calibrate on a dedicated
holdout with isotonic/Platt at production prevalence.

**Classifier (s3):** trained end-to-end for the first time —
`trees_ensemble`, self-reported test F1 0.915 / accuracy 0.92 across the five
classes. Read with the documented L1 caveat (its features embed detector
quantities that were fit on overlapping data); the number demonstrates the
stage *works*, not that it is unbiased.

**Inference (separate `infer` stage on the held-out 10% file):** 282 traces
scored with full audit trail (29 flagged, 0 truncated), RCA attribution
columns populated.

**Latency (idle-machine measurement inside eval):** p50 **32.7 ms**, p95
40.1 ms, p99 40.5 ms per trace, batch=1, CPU. The 20–30 ms target is a GPU
budget; the CPU path is already within 1.1–1.6× of it, and the device is
config (`runtime.device=gpu`).

**Position vs the product target (Recall ≥ 80% @ FPR ≤ 10%):**
overall recall@FPR≤0.10 is 0.40 (val) — the gap is driven, in order, by
(1) corpus size: ≈440 train-normal traces vs the spec's own 1e5–1e6
recommendation; (2) 10 training epochs (committed demo default, kept for
comparability — config exposes real values); (3) the semantics-free stand-in
embedder zeroing out the ipi/bias/hallucination classes, which are 60% of the
injected mix. On the EPI-detectable classes alone (mp+dpi) the detector is
already at 0.85 recall / FPR≤0.12.

## 3. Reviewer sanity checks

* **Does the threshold hold at realistic prevalence?** No — and the report now
  says so instead of hiding it: PPV at 1% prevalence is ~0.01–0.06 at the
  youden threshold chosen on injected-prevalence VAL. Production deployment
  must either (a) choose the threshold at the target FPR (the eval report
  publishes recall@FPR≤0.10 with its threshold), or (b) re-select on a VAL
  with production prevalence. Recorded as the F-22 operating guidance.
* **Where is the model vs the Recall ≥ 80% / FPR ≤ 10% target?**
  See §2 (SYNTH); on REAL the corpus is too small for the question to be
  meaningful. The gap driver ranking: (1) corpus size (spec itself demands
  1e5–1e6 traces; we trained on ≈770), (2) 10-epoch training (config default,
  raise on GPU), (3) stand-in embedder carries no semantics (OQ-3), so the SEM
  branch contributes structure-only signal.
* **Determinism**: same-seed bit-identity verified at baseline freeze and
  covered by cross-process tests; run manifests record seeds/config/data
  hashes for every validation run.

## 4. Metric-movement attribution (DECISIONS policy)

Every delta vs baseline is attributable: F-01 (semantic honesty: per-type AUCs
no longer oracle-inflated), F-03/F-23 (finite, per-element errors — absolute
error scales changed by design), F-04 (monotone calibration), F-02 (selection
on VAL), F-05 (bounded normalized features), F-40 (deep-experiment numbers
removed as invalid). No unexplained drift: the parity run (AUDIT_04) proved the
wiring alone was bit-identical before the fixes landed.
