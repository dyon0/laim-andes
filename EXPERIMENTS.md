# Detector experiments — the full catalogue

The detector trains one experiment per entry in `detector.experiments`
(UI parameter `experiments`, e.g. `["hub_mse_mse_08_4"]`), evaluates each on
VAL, and serves the winner. This file lists every available experiment, what
each knob means, and what to pick. Source of truth for the grid:
`ars/configuration/experiments/e2__detector.py`.

## The experiment code format

An experiment is named by a code the grid PARSES — you are not limited to
the catalogue below; any well-formed code works:

```
{epi}_{sem}_{comb}_{batch}_{lr}[_{arch}]
   │     │      │      │      │      └─ architecture: default | deep | wide
   │     │      │      │      └─ learning rate exponent: 4 -> 1e-4, 3 -> 1e-3, 5 -> 1e-5
   │     │      │      └─ batch size (traces per step): 08 | 16 | 32 | any integer
   │     │      └─ combined FMLP-AE loss: mse | hub (Huber, delta 1.0)
   │     └─ semantic LSTM-AE loss: mse | hub
   └─ EPI LSTM-AE loss: mse | hub
```

Example: `hub_mse_hub_32_4_deep` = Huber on EPI, MSE on semantic, Huber on
combined, batch 32, lr 1e-4, "deep" architecture.

## The three architectures

| Name | EPI LSTM-AE | SEM LSTM-AE | Combined FMLP-AE | Character |
|---|---|---|---|---|
| `default` | 2 layers (uni 64 → bi 32), latent 128 | 2 layers (bi 512 → uni 256), latent 768 | 3 layers (relu 128, tanh 32, relu 64) | The validated baseline. Fast, stable, fits small corpora. |
| `deep` | 10 alternating uni/bi layers (64→…→8→32→…→256), latent 128 | 14 alternating layers (768→…→24→256→…→512), latent 768 | 14-layer funnel (768→…→32→…→64) | Much higher capacity; needs large corpora and more epochs; slowest. |
| `wide` | growing widths (512→1024…, power-of-2 rounded), latent 128 | growing widths (2048→4096…), latent 768 | wide FMLP (768→…→512) | Widest layers, largest VRAM footprint per step; mid training cost. |

## The catalogue

`CODES` in `e2__detector.py` — the two uncommented entries are the shipped
default grid; the rest are pre-vetted combinations you can enable by listing
them in `experiments`:

| Code | Losses (epi/sem/comb) | Batch | LR | Arch | Status / notes |
|---|---|---|---|---|---|
| `hub_mse_mse_08_4` | hub / mse / mse | 8 | 1e-4 | default | **ACTIVE DEFAULT. Validation winner** — selected on VAL in the Phase 7 run; robust EPI loss tolerates heavy-tailed features. Start here. |
| `hub_mse_hub_32_4_deep` | hub / mse / hub | 32 | 1e-4 | deep | **ACTIVE DEFAULT.** High-capacity contender. Requires ≥ ~32 traces per split or it refuses to run (F-40 guard: batch 32 needs full batches). Historically its "youden 0.40" baseline number came from an UNTRAINED net (F-40) — distrust old reports about it. |
| `mse_mse_mse_08_5` | mse ×3 | 8 | 1e-5 | default | Plain-MSE baseline, very slow LR — for convergence-behavior comparisons, not production. |
| `mse_mse_mse_16_4` | mse ×3 | 16 | 1e-4 | default | Plain-MSE mid-range; a fair "no Huber" ablation. |
| `mse_mse_mse_32_3` | mse ×3 | 32 | 1e-3 | default | Aggressive LR; fastest wall-time, occasionally unstable losses (finiteness guard F-21 will abort rather than ship NaNs). |
| `hub_mse_mse_16_3` | hub / mse / mse | 16 | 1e-3 | default | Faster-training variant of the winner; try when training time is tight. |
| `hub_mse_mse_32_5` | hub / mse / mse | 32 | 1e-5 | default | Conservative LR at large batch; needs many epochs to move. |
| `mse_mse_hub_08_3` | mse / mse / hub | 8 | 1e-3 | default | Huber only on the combined head. |
| `mse_mse_hub_16_5` | mse / mse / hub | 16 | 1e-5 | default | Same, slow LR. |
| `mse_mse_hub_32_4` | mse / mse / hub | 32 | 1e-4 | default | Same, balanced settings. |
| `hub_mse_hub_08_5` | hub / mse / hub | 8 | 1e-5 | default | Double-Huber, conservative. |
| `hub_mse_hub_16_4` | hub / mse / hub | 16 | 1e-4 | default | Double-Huber, balanced — a reasonable third pick after the two defaults. |
| `hub_mse_hub_32_3` | hub / mse / hub | 32 | 1e-3 | default | Double-Huber, aggressive. |
| `mse_mse_mse_08_4_deep` | mse ×3 | 8 | 1e-4 | deep | Deep with small batch — the safest way to try `deep` on modest corpora. |
| `mse_mse_mse_16_3_deep` | mse ×3 | 16 | 1e-3 | deep | Deep, faster LR. |
| `mse_mse_mse_08_4_wide` | mse ×3 | 8 | 1e-4 | wide | Wide with small batch. |
| `mse_mse_mse_16_3_wide` | mse ×3 | 16 | 1e-3 | wide | Wide, faster LR. |
| `hub_mse_hub_32_4_wide` | hub / mse / hub | 32 | 1e-4 | wide | Wide counterpart of the deep default. |

Semantic-branch losses are `mse` throughout the catalogue: embeddings are
L2-normalized (bounded), so Huber's outlier resistance buys little there —
that is also why custom codes rarely need `hub` in the middle position.

## Selection metrics — set them explicitly

Each experiment picks its detection threshold and the grid picks its winner
by a metric. **Warning (documented in CLAUDE.md):** if you do not set them,
`build_grid` cycles `('youden', 'precision', 'recall', 'f1')` across
experiments BY POSITION in the list — experiment #1 gets `youden`,
#2 `precision`, and so on. That looks intentional but is historically
accidental. Always set both in the config/UI:

| Parameter | Meaning | Recommendation |
|---|---|---|
| `threshold_metric` | metric maximized when placing the anomaly threshold | `youden` (balanced TPR−FPR) as default; `precision` if false positives are expensive; `recall` if missed anomalies are expensive |
| `select_metric` | metric that ranks experiments against each other | same value as `threshold_metric` |
| `select_on` | split used for selection | keep `val` (selecting on test is leakage — fixed finding F-02) |

## Practical recommendations

1. **Production default**: `experiments=["hub_mse_mse_08_4"]`,
   `threshold_metric=youden`, `select_metric=youden`, `epochs=500`,
   `patience=50`. One experiment = shortest training; the winner of the
   honest Phase 7 validation.
2. **When you have a large corpus** (≥ tens of thousands of traces) and GPU
   time: add the capacity ladder —
   `["hub_mse_mse_08_4", "hub_mse_hub_16_4", "hub_mse_hub_32_4_deep"]`.
   Experiments run SEQUENTIALLY; training time scales linearly with the
   list length.
3. **Batch size floor**: an experiment refuses to run when a split has
   fewer traces than one full batch (F-40 guard — silently shipping an
   untrained model is what the legacy code did). On small/smoke corpora
   stick to `_08_` codes.
4. **Epochs**: the committed grid default is 10 (a legacy artifact — the
   original 500 was commented out, finding OQ-5); the platform UI default
   overrides it. For real training use 100–500 with `patience=50`; early
   stopping does the right thing from there.
5. **Custom codes** beyond the catalogue are legal and parsed on the fly
   (e.g. `hub_mse_mse_64_4` for batch 64 on a big corpus). Architectures
   are limited to the three named ones.
6. **Distrust pre-refactor numbers**: old reports' "~1200 MSE" figures used
   a different (timestep-normalized, unfloored) loss scale, and the deep
   experiment's old "youden 0.40" was an untrained network (F-40). Only
   post-F-23/F-40 runs are comparable.

## How to pass experiments on the platform

UI field `experiments` takes a JSON list:
`["hub_mse_mse_08_4", "hub_mse_hub_16_4"]`. The same list is available to
`run.py` as `--set 'detector.experiments=["hub_mse_mse_08_4"]'`. Per-branch
metrics, thresholds and the winner are reported in
`detector_metrics_holdout` / `eval_report`, and every experiment's training
curve lands in the run directory (`s2_meta.json`, `summary_report.html`).
