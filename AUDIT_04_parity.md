# AUDIT 04 — Refactor Parity Report (Phase 4)

## What changed in Phase 4 (behavior-preserving wiring)

* New `laim/` orchestration layer: typed config (`laim/config.py`, TOML + dotted CLI
  overrides), structured logging + crash-safe run manifest (`laim/runlog.py`),
  full evaluation surface (`laim/evaluation.py`), pipeline orchestration
  (`laim/pipeline.py`).
* Single entry point **`run.py`** with `synth | validate | prepare | train | eval |
  infer | all`. Training and inference are separate commands with separate
  artifacts (a trained run directory is the hand-off).
* Minimal `ars` change: `S2Config` gained three optional fields
  (`experiments`, `epochs`, `patience`, all defaulting to `None` = legacy values
  compiled into the experiment grid) so the committed demo debris (F-20) becomes
  configurable without touching numerics. `ars/main.py` is untouched and remains
  the deploy-platform adapter.

## Parity evidence

Same train file (`baseline/runs/real_a/train_spans.parquet`), same seeds, default
config (legacy values):

| Artifact | Legacy `ars.main` run | New `run.py train` run | Verdict |
|---|---|---|---|
| `hub_mse_mse_08_4/results.json` — per-epoch EPI/SEM/Combined losses, best_threshold, val/test metrics | `/tmp/mas-monitor#2026-08-12_11:01:53` | `runs/20260812_113507_train_b9a7aa977738` | **IDENTICAL** (bit-level) |
| `hub_mse_hub_32_4_deep/results.json` — same fields | same | same | **IDENTICAL** (bit-level) |

Characterization suite: 31 passed / 1 deselected after the wiring (`make test`).

## Deviations (all additive, none numeric)

1. The classifier stage's known crash (F-06) is now *recorded* in the run manifest
   (`train_classifier: failed: ValueError…`) instead of killing the run; the
   detector artifacts and evaluation complete. The crash itself is fixed in
   Phase 5, at which point the stage runs again.
2. New artifacts exist that the legacy path never produced: `manifest.json`,
   `run.log`, `eval_report.json` (per-type metrics, recall@FPR≤0.10,
   prevalence-adjusted PPV, ECE/Brier, latency percentiles).
3. Run directories live under `runs/` (config), not `/tmp/mas-monitor#<second>`.

## What Phase 4 deliberately did NOT do

No numeric fix has been applied yet — the P0 fixes land one-per-commit in Phase 5
with their characterization pins flipped in the same commit (PLAN.md step 4).
