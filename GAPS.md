# GAPS — derived from AUDIT_01

Two categories; engineering foundation first, because ML results produced on an
unvalidated, unobservable pipeline cannot be evaluated. Every gap traces to audit
findings (F-xx) — nothing here is a wishlist item.

## A. Engineering gaps

| # | Gap | Findings | What "closed" looks like |
|---|---|---|---|
| E1 | Dependency manifest & packaging | D-1, F-54 | `pyproject.toml` + pinned `requirements.txt`; `pip install -e .` works |
| E2 | Test infrastructure (none exists) | F-54 | pytest + fixtures + `make test` < 2 min CPU (Phase 3/6) |
| E3 | Unified config: file + CLI, no magic constants | F-29, F-20 | one typed config tree, TOML-overridable, every audit-flagged constant surfaced |
| E4 | Single entry point `run.py` with separated train / eval / infer | mission req., F-27 | `run.py synth/validate/prepare/train/eval/infer/all`; training and serving share no accidental state |
| E5 | Structured logging + run manifest | F-28 | every run writes `manifest.json` (config hash, data hash, seeds, versions, metrics, artifacts); logs are plain `logging` |
| E6 | Data validation gate in the load path | F-34, F-56 | spec-driven per-trace rejection wired into ingestion, strictness configurable, rejects logged |
| E7 | Typed error handling; no silent fallbacks | F-26 | bare excepts removed; degraded modes are explicit and logged |
| E8 | Artifact/checkpoint versioning | F-10, F-26 | artifacts carry schema version, embedder fingerprint, config hash; loaders verify |
| E9 | CI configuration | Phase 3 req. | workflow file running lint+tests on CPU |
| E10 | Report robustness (viz cannot kill training) | F-30, F-71 | chart rendering guarded; >5k-row frames handled; reports optional |
| E11 | Dead-code archival | F-51, F-52, F-53 | `legacy/` holds removed code with CHANGELOG notes |
| E12 | Deployment packaging script | F-31 | worker payloads generated from `ars/` at build time (documented; platform-side testing out of scope here) |

## B. ML / product gaps

| # | Gap | Findings | What "closed" looks like |
|---|---|---|---|
| M1 | Real semantic embeddings end to end | F-01 | all spans embedded (batched/cached/device-config); injector shares the same embedding fn; stub removed |
| M2 | Numerically safe normalization | F-05, F-03, F-21, F-23 | scale floors, winsorize-at-transform, latent std floor, finiteness guards, per-element loss convention |
| M3 | Honest model selection & thresholding | F-02, F-22 | select on VAL; threshold metric config (youden default); report recall@FPR≤10% + prevalence-adjusted PPV |
| M4 | Trustworthy calibration | F-04 | MAD floors, degenerate-fit fallback, monotonicity constraint, ECE/Brier in every train report |
| M5 | Leak-free splits | F-08, F-11, F-09 | group split by trace_id; selection stats train-only; max_len train-only + truncation policy |
| M6 | Contract-conformant features | F-07, F-50 | sentinel-aware aggregation; `duration_diff` fixed or removed; per-feature provenance |
| M7 | Working injector at scale | F-36 | u64-safe hashing; synthetic corpora of arbitrary size |
| M8 | Robust s3 stage (or explicit descoping) | F-06, F-12, F-32, F-33 | min-count guards + empty-frame safety (crash fix now); leakage-clean re-design documented as follow-up (OQ) |
| M9 | Full evaluation surface | F-24 | per-anomaly-type P/R/FPR/F1/ROC/PR-AUC, calibration, latency in the standard eval path |
| M10 | Latency path fit for near-real-time | F-27, F-35 | single encode pass, warmup, measured p50/p95/p99 per device in eval report |
| M11 | RCA seams (not RCA itself) | F-53 | per-span/per-feature reconstruction-error attribution exported per trace; counterfactual hook interface; s4 consumes the surface |
| M12 | Retraining/drift hooks | (product req.) | run manifests + frozen normalization stats make T+1 re-fit auditable; drift metrics deferred (recorded in PLAN as remaining work) |

## Explicitly out of scope (recorded, not built)

* RCA logic itself (only the attribution seams — team's R&D).
* SberDS platform-side integration testing (no platform access here).
* Real `deepvk/USER-bge-m3` weights (network-blocked; interface + pinning built and
  tested against the stand-in; OQ-3).
* s3 classifier's full nested-CV redesign (M8 long-term part) — the crash fix and
  guards land now; the redesign is specced in PLAN.md as remaining work.
