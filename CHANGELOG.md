# CHANGELOG

All notable changes on branch `claude/lumimas-anomaly-refactor-7stpda`
(legacy baseline: commit `e99d74e`). Finding IDs refer to AUDIT_01_findings.md.

## Fixed (P0s, each with the test that failed before / passes after)

- **F-36** injector text-noise crashed on ~5% of spans (u64 hash → i64 wrap in
  `str.slice`); synthetic corpora were impossible to build.
- **F-06** `Pad.split` crashed on empty frames; the s3 classifier now refuses
  starved classes with per-class counts instead of dying mid-pipeline
  (this pair is what made end-to-end runs fail on the shipped sample).
- **F-05** robust-scale explosion on quantile-degenerate features: scale floor
  (`scale_floor`, default 1e-2) + symmetric normalized-value clip
  (`norm_z_clip`, default 20), both persisted and applied at inference too.
- **F-23** branch losses divided by timesteps, not elements → EPI/SEM/Combined
  error magnitudes were incomparable (D-scaled). Per-element everywhere now.
- **F-03** latent std floor (`latent_std_floor`) — degenerate latent dims no
  longer produce 1e8-scale inference inputs (baseline: 3.9e8 test error).
- **F-21** non-finite train/val losses abort with `FloatingPointError`;
  inference refuses to emit NaN/Inf scores.
- **F-08** split unit is now the trace (all agent-rows travel together);
  anomaly classes processed in sorted order (no group-iteration-order concat).
- **F-11** feature-selection statistics (fill/static/correlation/max-abs) fit
  on planned train-normal traces only; hard runtime check that the planned
  split equals the actual one.
- **F-09** `max_len` fits on train only (+ optional cap); over-length traces
  follow `truncation_policy` and are flagged `detector_truncated` at inference.
- **F-07** sentinel `-1` values no longer pollute features: `llm_tokens`
  treats unknown usage as null (spec §7), `tool_compression` uses null for
  non-tool spans.
- **F-50** `duration_diff` was identically zero ((end−start)−(end−start));
  now the step-to-step duration delta.
- **F-02** experiment selection reads VAL (`select_on`, both s2 and s3);
  best_info records the selection value and the honest test value separately.
- **F-04** calibration guards: MAD floors (abs+rel), min val class counts,
  weight cap, and a hard `w_comb > 0` monotonicity requirement with fallback
  to fixed priors (baseline had `w_comb = -209` → anti-calibrated p_anomaly).
- **F-01** semantic embeddings are real for every span (the committed code
  stubbed all normal spans to zero vectors, so the SEM branch detected the
  fact of injection, not semantics); one embedder instance shared with the
  injector.
- **F-10** embedder fingerprint recorded in S1Meta and verified before serving.
- **F-34** data-contract validation gate (`off`/`warn`/`strict`) wired into
  `prepare`; strict mode trains only on conformant traces.
- **F-30** chart rendering can no longer kill training (altair row cap
  disabled; render failures degrade to missing images).

## Added

- `run.py` — single entry point (`synth | validate | prepare | train | eval |
  infer | all`); training and inference are separate commands and artifacts.
- `laim/` — typed TOML+CLI config, structured logging, crash-safe run
  manifests (config hash, data fingerprints, seeds, stage timings, metrics).
- `laim/evaluation.py` — per-anomaly-type P/R/FPR/F1/ROC/PR-AUC, recall @
  FPR≤target, prevalence-adjusted PPV, ECE/Brier + reliability, latency
  p50/p95/p99.
- `ars/models/m2__detector/attribution.py` — RCA seams (M11): per-span and
  per-feature reconstruction-error attribution + counterfactual hook; exposed
  through `detect_anomalies(attribution_top_k=…)`; s4 formats it into
  `rca_report_str` (RCA logic itself remains future R&D).
- Packaging: `pyproject.toml`, pinned `requirements.txt`, `Makefile`, CI
  workflow; test suite (characterization, contract, anti-leakage, numerics,
  attribution, embeddings).
- `baseline/` — frozen legacy baseline (metrics, determinism check, stand-in
  embedder builder).

## Added (SberDS deployment)

- Root `descriptor.json` — one dual-mode platform node (`mode = train |
  inference`): train emits a portable model bundle (zip on shared storage,
  path on `model_out`); inference consumes it (`model_in` port or
  `model_path`), scores spans with the full audit trail and emits the
  legacy-compatible product contract (`anomaly_traces`, `test_anomalies`).
  GPU/CPU via the `device` select on the `py312-gpu` image; all key config
  surfaced as UI parameters + a `config_overrides` escape hatch.
- `laim/platform.py` — the adapter behind `run.py::main(**params)`: UI-param →
  config mapping, embedder zip resolution, bundle create/resolve (paths stored
  bundle-relative, absolutized on extraction; embedder fingerprint verified,
  training embedding params re-applied at inference), product-contract
  builder reusing the legacy `ars.main` enrichment.
- `tests/test_platform.py` — descriptor↔adapter contract pins, param mapping,
  bundle round-trip, and a slow end-to-end train→bundle→inference test through
  the exact platform entry point.
- `deploy/README.md` — deployment guide; legacy `deploy/descriptors/` +
  `deploy/nodes/` (codebase-tarball pattern) documented as obsolete.

## Fixed (SberDS deployment, from the environment probe of 2026-08-13)

- `requirements.txt` is now the platform install manifest: torch removed (the
  `py312-gpu` image preinstalls a working `2.8.0+cu128`; a bare torch pin let
  the mirror's `+xpu` build shadow it — the `libsycl.so.9` crash), the JAX GPU
  stack repinned from cu13 to `jax-cuda12-plugin==0.11.0` (driver 570.x cannot
  run CUDA 13), `numpy/pandas/pyarrow/tqdm` aligned to the image so pip leaves
  the platform's own interpreter dependencies untouched. Verified: fast suite
  (87 tests) passes on the exact new stack in a clean venv (D-3).
- Dataframe ports arrive as a DIRECTORY of `part-*.snappy.parquet` (observed:
  100 parts / 56 GB): `spans_scan_source()` normalizes file|dir|glob once, in
  every consumer (validate, gate, prepare, infer, product contract);
  `file_fingerprint()` handles directories/globs (name+size manifest hash) and
  samples files > 2 GB instead of reading tens of GB at run start.
- Model ports arrive as an extension-less blob (`unstructured_data`, ZIP by
  magic bytes): `_resolve_embedder` now sniffs zip/tar content and locates the
  model root by `config.json`, instead of trusting a `.zip` suffix.
- `model_store_dir` unwritable (probe: `/mnt/data` permission-denied) no longer
  kills a finished training run: writability is checked up front with a /tmp
  fallback and a loud warning (OQ-6 tracks the real shared-storage fix).
- `run_node` caps `POLARS_MAX_THREADS`/`OMP_NUM_THREADS` to the cgroup CPU
  quota (host reports 128 CPUs; the container quota was 8 — polars would
  oversubscribe 16×).
- `cmd_prepare` memory preflight: warns when input size × the measured
  in-memory multiplier exceeds the cgroup memory limit, before the OOM kill.
- Probe node: survives ports delivered as in-memory pandas DataFrames
  (observed in the probe run; reported instead of crashing section 11) and no
  longer spends 6×120 s on the mirror's hanging `pip index versions`.

## Archived to `legacy/` (never deleted without a trace)

- `verification.py` (was `ars/tools/reproducibility/`) — orphan module, zero
  importers; its repr-based frame comparison also passes on truncated output.
- `code_format.py` (was `ars/tools/utilities/`) — developer source-alignment
  tool, unused by the pipeline; its read-only CLI mode crashes on unpacking.
- `s1__data_dead_blocks.py.txt` — two commented-out function bodies carried
  inside `ars/stages/s1__data.py` (an older `fill_missing_values` and the
  transformers-based `make_embedder`).

## Removed (empty files, nothing to archive)

- `ars/tools/utilities/codebase_check.py` (0 bytes), `ars/configuration/
  c0__paths.txt` (0 bytes), `ars/configuration/c4__rca.py` (0 bytes),
  `ars/configuration/c5__reports.py` (0 bytes), two zero-byte Russian-named
  `todo__…` marker files.

## Knowingly retained (documented, not touched)

- `ars/tools/abstraction/` composition DSL — still used by `ars/main.py`'s
  legacy pipeline composition; two vendored copies remain under
  `deploy/nodes/*` because the deploy platform flattens source trees.
- Unused helpers inside `ars/tools/visualisations/viz.py` and the unused
  recursion schemes — live modules, dead branches; candidates for a later
  sweep once the deploy payloads are regenerated.
