# FINAL REPORT — LAIM / LumiMAS anomaly-detector overhaul

Branch `claude/lumimas-anomaly-refactor-7stpda` · legacy baseline `e99d74e` ·
written for a technical reader who has not watched the work.

Environment reality: this ran in a CPU-only container (4 cores, 15 GB) with
huggingface.co network-blocked. The 2×H100 target hardware and the real
`deepvk/USER-bge-m3` embedder were unavailable; consequences are stated inline
and in OPEN_QUESTIONS.md (OQ-2, OQ-3). Everything is reproducible from the
committed configs and seeds.

---

## 1. What was wrong (evidence first)

The full inventory is `AUDIT_01_findings.md` (14 P0, 16 P1, with file:line and
proof tests). The load-bearing five:

1. **The pipeline did not run.** Two independent crashes: the classifier
   stage died on any small corpus (`Pad.split` on an empty stratified split —
   this killed the shipped 43-trace sample), and the injector died on any
   *large* corpus (u64 hash wrapping negative in `str.slice`, ~5% of spans).
   There was **no corpus size at which end-to-end training completed.** The
   frozen baseline (AUDIT_00) documents both.
2. **The semantic branch was fake.** Every normal span's embedding was a
   hard-coded zero vector (`s1__data.py:504`), while the anomaly injector
   embedded its corrupted texts with a real model. The SEM branch therefore
   detected *whether the injector had touched a trace* — hallucination AUC 1.0
   was an oracle artifact, and ipi/bias injections were mathematical no-ops
   (their manifold shifts were scaled by the std of zeros). Every previously
   reported semantic/combined metric is invalid.
3. **The numbers that looked like metrics were numerics failures.** Combined
   test error 392,328,128 (division by `std+1e-8` on constant latents);
   branch "MSE ~1200" magnitudes were D-scaled loss bookkeeping on top of a
   robust scaler that amplified quantile-degenerate features to |z|=42;
   calibration weights fit on 6 val points came out `w_comb = −209` — **higher
   reconstruction error lowered p_anomaly** (ECE = Brier = 0.80).
4. **Evaluation could not be trusted structurally.** Model selection read the
   test set (s2 and s3); the s3 classifier trained on the detector's VAL∪TEST
   with features shaped by val-fitted thresholds/calibration; feature selection
   was fit on the full pre-split corpus; the split unit allowed one trace's
   agent-rows into different splits; `max_len` was sized on test data.
5. **A model that trained nothing shipped metrics.** Found *during* Phase 7 by
   the new finiteness guard: with batch 32 > 18 train traces, epochs contained
   zero batches, losses were `[nan, …]`, early stopping never fired, and the
   loop returned the untrained init — which the legacy code evaluated
   (test youden 0.40) and made eligible to win model selection (F-40). The
   frozen baseline's deep-experiment artifacts prove it.

Also material: sentinel `-1` values fed into features against the data spec's
explicit rule; a dead always-zero feature (`duration_diff`); no input
validation anywhere in the train path despite a complete unused spec-driven
validation engine; no dependency manifest; zero tests; unpinned embedder;
reports (altair) able to crash training after models were fitted.

## 2. What changed, and what each change bought

Every fix landed as its own commit with the characterization pin flipped in the
same commit; `CHANGELOG.md` is the ledger. In metric terms:

| Change | Bought |
|---|---|
| F-36/F-06 crash fixes | End-to-end runs exist at all. First-ever full run on a realistic corpus completed (950 traces, 5 stages, ~55 min CPU). |
| F-01 real embeddings + shared injector embedder | Honest per-type structure: hallucination AUC 1.0→0.67 on the fixture (oracle removed); ipi/bias now *measurably* at chance instead of silently undetectable. |
| F-03/F-05/F-23 numeric floors + per-element loss | Combined test error 3.9e8 → 0.126 (finite, comparable across branches); normalized features bounded (max |z| 42 → ≤20). |
| F-04 calibration guards | `w_comb` −209 → guaranteed positive (monotone p_anomaly); ECE 0.80 → 0.25 (real corpus) — still weak, now *measured* (see §6). |
| F-02 selection on VAL | Test metrics are a holdout again; best_info records selection vs holdout values separately. |
| F-08/F-09/F-11 split & selection hygiene | No trace spans two splits; selection stats and max_len derive from train only; proof tests in `test_anti_leakage.py`. |
| F-07/F-50 feature correctness | Sentinels no longer bias aggregates (spec §7); `duration_diff` carries signal instead of zeros. |
| F-40 zero-batch guard | Untrained models can no longer masquerade as experiments. |
| F-34 validation gate | Contract violations are rejected/logged before training (off/warn/strict). |
| F-10 embedder fingerprint | A swapped embedding model now fails loudly at serve time instead of silently invalidating checkpoints. |

## 3. What was built

- **`run.py`** — one entry point: `synth | validate | prepare | train | eval |
  infer | all`. Training and serving are separate commands with separate
  artifacts; `infer` takes a finished run directory and scores any spans file
  with a full audit trail (every trace scored + flagged, truncation flags, RCA
  columns).
- **`laim/`** — typed config (defaults ← TOML ← `--set a.b=c`), structured
  logging, crash-safe run manifests (config hash, input SHA-256, seeds,
  package/git provenance, stage timings, metrics). Every key parameter the
  audit found hardcoded is config now.
- **Evaluation surface** (`laim/evaluation.py`): per-anomaly-type
  P/R/FPR/F1/ROC/PR-AUC, recall @ FPR≤target, prevalence-adjusted PPV,
  ECE/Brier + reliability tables, latency percentiles — produced by every
  `eval` run as `eval_report.json`.
- **RCA seams** (audit said build seams, not RCA): per-span and per-feature
  reconstruction-error attribution + a counterfactual hook
  (`ars/models/m2__detector/attribution.py`), surfaced as `rca_*` columns and
  a human-readable `rca_report_str` in s4. Tests prove the surface localizes
  a poisoned span/feature.
- **Packaging + CI + tests**: pyproject, pinned requirements, Makefile,
  CI workflow, and a 74-test suite (see §4).
- The legacy `ars.main.main` deploy contract is untouched and parity-proven
  (`AUDIT_04_parity.md`: new runner reproduced legacy losses/thresholds/metrics
  bit-for-bit before any fix landed).

## 4. Test coverage — what is verifiable now

`TEST_REPORT.md` has the full matrix. Summary: 74 tests across
characterization (golden pins at fixed seed), data contract (10 violation
classes + whole-trace rejection + recast), property-based (Hypothesis:
injector totality, recall@FPR bound, normalization round-trip, permutation
invariance), anti-leakage (split/scaler/selection/threshold), numerics
(overfit-one-batch, all-sentinel adversarial, NaN aborts), determinism
(bit-identical across processes), model behavior (error separation +
monotonicity in perturbation strength), calibration (monotone p, priors
fallback), integration (raw spans → p_anomaly on CPU), performance, failure
modes (typed errors for every guard). `make test` ≈ 2 min warm on CPU.

Honestly untested: full-scale GPU training, the real embedder, the s3
stack's statistical validity (guards tested; biases documented as L1–L12),
deploy-platform integration. Reasons in TEST_REPORT.md.

## 5. Metrics — baseline vs now

The only fully honest sentence: **the legacy numbers were not measurements.**
Where they existed they were produced by an oracle (F-01), an overflow (F-03),
an untrained network (F-40), or test-set selection (F-02). Side-by-side tables
with per-type detail: `VALIDATION.md`. Headlines:

- Real 43-trace corpus: end-to-end completes; errors finite; calibration
  monotone; ROC-AUC 0.64 (down from the baseline's partly-oracle 0.80 — a
  regression in the number, an improvement in the truth).
- Synthetic 950-trace corpus (previously impossible to run): detector recall
  0.40 @ FPR 0.10 (val), per-type mp 0.91 / dpi 0.81 / hallucination 0.15 /
  ipi 0.21 / bias 0.12 (test); classifier trains and self-reports F1 0.92
  (biases documented); latency p50 32.7 ms/trace CPU vs the 20–30 ms GPU
  target; per-trace RCA attribution produced.
- Gap to Recall≥80%/FPR≤10%, ranked: corpus size (≈440 train-normal traces vs
  the spec's 1e5–1e6), 10-epoch demo default (config now), semantics-free
  stand-in embedder (kills ipi/bias/hallucination detectability — 60% of the
  injected mix). On the EPI-detectable classes (mp+dpi) the detector already
  reaches ~0.85 recall at FPR≈0.12.

## 6. What is still broken or risky (ranked)

1. **Calibration quality** (not correctness): p_anomaly is monotone and
   bounded but poorly calibrated (ECE 0.47/0.65 on synth val/test). Needs a
   dedicated calibration holdout at production prevalence (isotonic or Platt)
   — the eval surface to measure it now exists.
2. **s3 classifier leakage chain** (L1–L12, F-33): guards landed (no crashes,
   min-count, selection on val), but its train pool is still the detector's
   VAL∪TEST with detector-shaped features. Its metrics must be read as
   optimistic until the nested-split redesign (specced in PLAN.md) lands.
3. **Real embedder unvalidated** (OQ-3): all semantic-branch conclusions here
   are lower bounds; swap in the fingerprint-pinned `deepvk/USER-bge-m3` on
   network-enabled hardware and re-run Phase 7 (one command per corpus).
4. **Threshold transfer at production prevalence** (F-22): quantified
   (PPV ≈ 0.03–0.04 at 1% prevalence at the youden threshold). Operating
   guidance: select the threshold at the FPR target from the eval report, or
   re-select on production-prevalence VAL.
5. **GPU determinism and latency** unmeasured (no GPU here); CPU path is
   verified deterministic and 32.7 ms p50.
6. **Deploy tree incomplete** (F-31): worker payloads
   (`deploy/nodes/_codebase/data/`, incl. `run_all.py`) are not in the repo and
   `deploy.py` is empty — packaging must be rebuilt before any platform deploy.
7. Residual P2 debris: TUI/monkey-patching observability layer still exists
   under the legacy stages (the laim layer supersedes it operationally);
   composition-DSL copies vendored under `deploy/`.

## 7. Open questions and defaults taken

See `OPEN_QUESTIONS.md`: OQ-1 branch naming (harness branch used), OQ-2 no GPU
(CPU-sized runs, hardware recorded in every artifact), OQ-3 embedder blocked
(deterministic stand-in + fingerprint pinning; real-model swap is a config
change), OQ-4 baseline corpus choice (both real + synthetic, provenance
recorded), OQ-5 epochs=10 demo debris (kept for baseline comparability; config
now controls it). No question required deviating from the conservative default.

## 8. Suggested next steps (in order)

1. On GPU hardware with network: download + fingerprint `deepvk/USER-bge-m3`,
   rerun `run.py all` on a ≥1e5-trace corpus with `detector.epochs=500`,
   `patience` real; compare against `VALIDATION.md`.
2. Calibration holdout + isotonic recalibration at target prevalence.
3. s3 nested-split redesign (PLAN.md remaining work).
4. Regenerate deploy payloads from `ars/` + `laim/`; wire `run_all.py` to
   `ars.main.main` (contract already parity-proven).
5. Feed the RCA seams (per-span attribution + counterfactual hook) into the
   team's RCA R&D.
