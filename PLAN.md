# PLAN — sequenced execution

Ordering rationale: (1) you cannot refactor safely without a safety net; (2) you
cannot evaluate ML fixes without a runnable pipeline and an honest eval surface;
(3) P0 fixes go in dependency order so each one's effect is measurable in
isolation; (4) each step is one commit (or a small series), each has tests, each is
revertible by `git revert` without touching neighbors.

## Step 0 — Packaging (E1) ✅ unblocks everything
`pyproject.toml`, pinned `requirements.txt`, `Makefile` (`make test`, `make train-smoke`).
Rollback: trivial.

## Step 1 — Phase 3 safety net (E2)
* Tiny deterministic fixtures in-repo: generator-built spans covering normal traces,
  each anomaly class, malformed spans (each spec violation class), edge cases
  (single-span trace, deep tree, missing session_id, all-sentinel LLM fields).
* Characterization tests pinning current behavior (feature matrix hash, EPI
  normalization params, split membership, threshold, calibration params) at fixed
  seed on the fixture corpus — CPU, seconds. Tests that pin known-bad behavior are
  marked `@pytest.mark.characterization_bug(finding='F-xx')`.
* CI workflow file.
Unblocks: every subsequent change. Rollback: delete tests (never needed).

## Step 2 — Config + entry point (E3, E4), behavior-preserving
One typed config tree (dataclasses, TOML file + CLI dotted overrides). `run.py`
with subcommands `synth | validate | prepare | train | eval | infer | all`.
Legacy `ars.main.main(**ui)` kept as a thin adapter over the new config (deploy
contract). No numeric behavior change (characterization green).
Rollback: revert commit; legacy entry still works throughout.

## Step 3 — Logging + run manifest (E5), typed errors (E7)
`logging` everywhere the TUI printed; run manifest JSON per run; bare excepts
replaced. Behavior-preserving.

## Step 4 — P0 fixes, dependency order (M2, M5, M7, part of M8)
Each with the proof test from AUDIT_01, characterization updated in the same commit:
1. **F-36** injector u64 hash fix → synthetic corpora become possible (needed by
   every later validation step).
2. **F-06** s3 min-count guard + `Pad.split` empty-frame safety → end-to-end runs
   complete.
3. **F-05** scale floor + winsorize-at-transform; **F-23** per-element loss.
4. **F-03** latent std floor; **F-21** finiteness guards.
5. **F-08** split by trace_id; deterministic group order.
6. **F-11** feature selection train-only.
7. **F-09** max_len train-only + explicit truncation policy.
8. **F-07** sentinel-aware features.
9. **F-02** selection on VAL.
10. **F-04** calibration guards + fallback + monotonicity.
11. **F-01** real embeddings both paths (stand-in model here; interface identical
    for the real model) + **F-10** embedder fingerprint pinning.
Ordering note: numeric fixes (3–4) precede split/selection fixes so metric movement
is attributable; F-01 lands late because it changes the data most.

## Step 5 — Evaluation module (M3, M9): per-type metrics, ROC/PR, ECE/Brier,
recall@FPR≤10%, prevalence-adjusted PPV, latency p50/95/99. Used by `run.py eval`
and by training reports.

## Step 6 — Validation gate (E6) wired into `prepare`, strictness configurable.

## Step 7 — Inference path (M10): single encode pass, full audit trail (all traces
scored), warmup, measured latency. `run.py infer`.

## Step 8 — Artifact versioning (E8) + report robustness (E10).

## Step 9 — Dead-code archival (E11) into `legacy/` + CHANGELOG.

## Step 10 — RCA seams (M11): per-span & per-feature reconstruction-error
attribution exported; counterfactual hook interface; s4 consumes it (still no RCA
logic).

## Step 11 — Phase 6 comprehensive test suite (all 12 layers), TEST_REPORT.md.

## Step 12 — Phase 7 validation run (synthetic corpus at scale + real sample),
side-by-side vs baseline, VALIDATION.md.

## Step 13 — Phase 8 FINAL_REPORT.md + CLAUDE.md.

## Metric-movement policy
Steps 4.3–4.11 intentionally move metrics. Each lands with before/after numbers in
the commit message, attributed to its finding; DECISIONS.md gets a summary table.
Anything that would change *published* numbers (the historical "~1200 MSE" reports,
pilot results) is flagged in OPEN_QUESTIONS.md — the conservative default is that
old artifacts remain readable (loader keeps backward compat) but new runs use the
fixed path.

## Active threads (as of 2026-08-14 — read this first in a new session)

The platform pipeline WORKS END-TO-END (operator-confirmed). The current
frontier is DETECTION QUALITY on the operator's real corpus (3216 traces,
single agent, no real labels — metrics measure injected anomalies):

* First 500-epoch run: ROC AUC 0.54, PR AUC 0.36 at ~33% test prevalence —
  near-chance ranking; threshold tuning cannot help until the score improves.
* Diagnosis (see the 2026-08-14 session): (a) 87% of traces violate the data
  contract and trained anyway (gate=warn); (b) the selected EPI feature set
  collapsed to ~57 variants of `avg_word_length_sem_*` (single-agent corpus +
  max_correlation=0.999); (c) metrics average 5 injected classes of which
  ipi/bias were near-chance even in controlled validation — check
  `eval_report.test.per_anomaly_type` before concluding anything.
* Agreed next experiment (parameter-only): validation_gate=strict vs warn,
  min_fill_rate=0.3, max_static_rate=0.99, max_correlation=0.9,
  embedding_max_length=1024, epochs=500/patience=50, experiments=
  ["hub_mse_mse_08_4","hub_mse_hub_16_4","hub_mse_hub_32_4_deep"],
  threshold_metric=select_metric=youden, classifier_enabled=false while
  iterating. NEVER threshold_metric=recall (degenerates to flag-everything).
* Planned feature work (OQ-8): expose `data.injection_fractions` (enables
  hallucination-focused detection + realistic-prevalence studies) and wire
  the dead `scale_floor`/`norm_z_clip` knobs.
* Key semantics to not re-derive: norm_*/anom_* ratios are SPLIT fractions
  (anomalies never enter training); the classifier (s3) is downstream of
  detection and cannot affect detector metrics; injected share is 20% of
  traces via `Plan.fractions`.

## Remaining work (not in this engagement's budget, recorded honestly)
* s3 nested-CV redesign (M8 long-term); currently guarded, biases documented.
* Drift metrics / retraining automation (M12) beyond the manifest hooks.
* Real embedder download + re-validation on GPU hardware (OQ-2/OQ-3) —
  partially closed: the real `deepvk/USER-bge-m3` served both train and
  inference on the platform; QUALITY re-validation (metrics with real
  semantics vs the stand-in numbers) remains open.
* ~~Deploy platform integration testing (E12)~~ — DONE 2026-08-14: operator
  confirmed the full train → bundle-over-port → inference pipeline works
  end-to-end on SberDS (8×H100 node; real corpus; classifier included).
* Out-of-core s1 (streaming/chunked prepare + scoring) — REQUIRED for the
  56 GB+ production corpora (the eager pipeline peaks at ~12-25x on-disk
  size; see deploy/README.md sizing). Until then: sampled training +
  chunked scoring.
* Multi-GPU detector training (experiment grid across GPUs). Embedding —
  the dominant GPU cost — is data-parallel since D-4; the s2 autoencoders
  are small and train on one GPU, so grid parallelism is a wall-clock
  optimization. CONSTRAINT (D-4 addendum): the SberDS wrapper is not
  spawn-safe and fork is unusable after CUDA init, so "one worker process
  per GPU" cannot be done naively in-node — viable shapes are separate
  platform nodes per experiment or a forkserver started before CUDA.
