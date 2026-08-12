# TEST REPORT (Phase 6)

`make test` — fast CPU suite; `make test-all` adds the slow layer (micro-training,
integration, cross-process determinism, latency). No test requires a GPU or
network access; the stand-in embedder is built on demand from
`baseline/build_standin_embedder.py`.

Suite size: **74 tests** (68 fast + 6 slow). Fast wall-clock: ~2 min warm
(first run after a container restart pays JAX compile, ~3.5 min).
The integration test also caught a real config bug while being written
(list-valued CLI overrides were split into characters) — fixed with
`tests/test_config.py` pinning the behavior.

## Layer coverage

| Layer (mission spec) | Where | Notes |
|---|---|---|
| Unit | `test_characterization_s2.py` (masked loss, threshold sweep, Pad shapes), `test_attribution.py` (top_k), `test_embeddings.py` (fingerprint), `test_property_based.py` (metric arithmetic) | pure functions pinned with exact values |
| Data contract | `test_data_contract.py` | per-field validity for 10 violation classes, whole-trace rejection, sentinel constants, 46-field schema vs the shipped sample, recast round-trip + missing-column sentinel fill |
| Property-based (Hypothesis) | `test_property_based.py` | recall@FPR never exceeds target; PPV∈[0,1]; ECE≈0 under perfect calibration; confusion adds up; injector text-noise total over arbitrary unicode/seeds (F-36 property); normalization round-trip within clip bounds; span-permutation invariance of features |
| Invariant / anti-leakage | `test_anti_leakage.py` | no trace_id in two splits; train has zero anomalies; normalization params immune to val/test perturbation; feature selection immune to non-train corruption; threshold-selection source pinned |
| Numerical | `test_numerics_and_behavior.py` | overfit-one-batch collapse (structured data); all-sentinel LLM span → finite features; NaN input aborts training (F-21); latent std floor bounds (F-03) |
| Determinism | `tests/golden/` characterization (same-process, bit-level) + `test_micro_train_bit_identical_across_processes` (slow, two fresh interpreters) | verified bit-identical |
| Model behavior | `test_numerics_and_behavior.py` | reconstruction error ≥3× higher on perturbed traces; error monotone in perturbation strength; `test_attribution.py`: attribution localizes the poisoned span/feature |
| Calibration | `test_characterization_s2.py` (MAD floors, priors fallback, w_comb>0), `test_numerics_and_behavior.py` (p∈(0,1), monotone in error) | ECE bound asserted in integration report range |
| Integration | `test_integration.py` (slow) | raw AEF spans → prepare → train (1 epoch) → eval → infer on CPU; asserts manifest stages, eval-report shape, full audit trail, RCA columns, p∈(0,1) |
| Regression | `tests/golden/golden.json` pins features/selection/split/normalization/micro-train at seed 12345; regenerated only with an explained diff (`make golden`) | post-Phase-7 metric pins recorded in VALIDATION.md |
| Performance | `test_branch_inference_latency_budget` (slow) | p50 < 100 ms per trace on CPU for the micro model; the 20–30 ms product target is a GPU number — see honest gaps below |
| Failure modes | `test_numerics_and_behavior.py` (strict gate raises on non-conformant corpus, infer without model dir), `test_embeddings.py` (fingerprint mismatch raises), `test_characterization_s2.py`/`s1` (typed errors) | every failure is a typed exception with an actionable message |

## Characterization discipline

Known-buggy behavior was pinned with `@pytest.mark.characterization_bug` +
finding ID during Phase 3; every Phase 5 fix flipped its pin in the same commit
(see git history: F-36, F-06, F-05, F-23, F-03, F-04, F-07, F-50 pins all
converted to fixed-behavior tests). No failing assertion was deleted or
loosened; two test premises were corrected and documented in commits
(polars concat dtype in a fixture; overfit-one-batch on noise → structured data).

## Honestly untested, and why

* **Full-scale training quality** (real epochs, 10⁵–10⁶ traces): needs the
  target GPU hardware and the real `deepvk/USER-bge-m3` weights (network-blocked
  here, OQ-2/OQ-3). The Phase 7 validation run covers a CPU-scale surrogate.
* **The 20–30 ms latency target**: it is a reference-GPU budget; this container
  has no GPU. CPU latency is measured and reported instead (eval reports +
  perf test with a CPU-scale bound).
* **GPU determinism / multi-GPU**: no GPU available; `gpu` marker reserved.
* **The s3 classifier stack's OOF internals** (F-33 fixes are future work, see
  PLAN.md remaining): its crash-guards are tested; its statistical quality is
  explicitly not certified.
* **Deploy-platform integration** (`deploy/` nodes): the platform is not
  reachable from this environment; packaging remains documented-only (F-31).
* **`ars/main.py` legacy adapter end-to-end**: exercised indirectly (its stages
  are the same functions the new pipeline calls; parity proven in AUDIT_04),
  but the `/tmp/mas-monitor#…` script path itself has no automated test — it
  exists only for the deploy platform's `run_all.py` contract.

Line coverage was deliberately not chased (weak signal on an ML repo);
the table above is the coverage claim.
