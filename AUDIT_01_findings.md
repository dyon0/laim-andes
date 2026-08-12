# AUDIT 01 — Deep Audit Findings (Phase 1)

Legacy code as of commit `e99d74e`. Line numbers refer to that commit.
Severity: **P0** wrong results / silent corruption · **P1** blocks reliability or
scaling · **P2** maintainability · **P3** nice to have.

Each finding: location · why it is wrong · blast radius · proposed fix · **proof**
(the test/measurement that fails before the fix and passes after).

Cross-references: `B-*` = already sketched in AUDIT_00; `L-*` = classifier-stack
findings enumerated during the parallel audit of `ars/models/m3__classifier/*`.

---

## P0 — wrong results / silent corruption

### F-01 · Semantic branch is an injection oracle (zero-stub vs real embedder)
**Where.** `ars/stages/s1__data.py:504-505` (stub `return df.with_columns(pl.lit((0.0,)*1024)…)`
before unreachable real code); `s1__data.py:1029-1033` (injector receives a *real*
embedder via `make_embedder`); `ars/data/anomalies_injection.py:712-728`
(hallucination replaces `sem_vector` with real embeddings), `:590-594` +
`Manifold.fit` `:350-357` (manifold scale fit on zero vectors = 0 ⇒ dpi/ipi/mp/bias
semantic shifts are exact no-ops).
**Why wrong.** Every normal span's embedding is the constant zero vector; only
hallucination-injected spans carry real (unit-norm) vectors. The SEM AE learns to
reconstruct zero, so `e_sem > 0 ⇔ trace was hallucination-injected`. The semantic
detector detects the *injection bookkeeping*, not semantics. Simultaneously IPI and
Bias (which per `OperatorSpec` touch only semantics, `anomalies_injection.py:174-178`)
become **undetectable by construction**.
**Blast radius.** All published SEM/Combined metrics; all per-type recalls; the
calibration (F-04); every trained checkpoint.
**Evidence.** Baseline: `sem_recon_error_mean` = 0.0 exactly on train/val, nonzero
on test only via the hallucination trace; `sem_latent_std = 1e-8` for all 768 dims.
**Fix.** Delete the stub; compute real embeddings for all spans (batched, cached,
device-configurable); pin the model revision (F-10). Injector must use the *same*
embedding function as s1.
**Proof.** Test: `sem_vector` of a normal trace must be text-dependent (two spans
with different `output_text` ⇒ different vectors); integration: per-type AUC on the
synthetic corpus — hallucination AUC drops from ~1.0 (oracle) to a real value, and
ipi/bias AUC rises above 0.5 once semantic shifts actually apply.

### F-02 · Model selection reads TEST
**Where.** `ars/stages/s2__detector.py:535-538` (`_pick_best` maximizes
`test_metrics[select_metric]`), `:550`, `:564-574` (best_info + S2Meta record test
numbers as the headline); same pattern in s3: `s3__classifier.py:234-237,277-281`.
**Why wrong.** The winning experiment is chosen on the split that is then reported
as the holdout ⇒ winner's-curse bias; TEST is no longer a holdout for either stage.
**Blast radius.** Every reported "best" metric; experiment ranking.
**Fix.** Select on VAL (`val_metrics`), report TEST for the selected model only.
**Proof.** Anti-leakage test asserting the selection function receives only
val metrics; regression: report both numbers, expect selected-model TEST ≤ old
cherry-picked TEST on average.

### F-03 · Latent normalization divides by ≈0 (`std + eps`), overflow at inference
**Where.** `ars/stages/s2__detector.py:306-310` (`std = jp.std(...) + eps`, eps=1e-8);
consumed at `:397-404`, `confidence.py:166-167`.
**Why wrong.** A degenerate latent dimension (constant across train — guaranteed by
F-01 for all SEM dims, but possible for any dead unit) yields std=0 ⇒ divisor 1e-8 ⇒
inference z-values ~1e8 ⇒ combined error overflow (baseline: 3.9e8 mean test error),
saturated calibration logits, and float32 overflow risk (inf at ~3.4e38 is reachable
with squared errors of 1e8-scale inputs: (1e8)²=1e16 per dim, sum over 896 dims
≈ 9e18 — still finite, but one more magnitude in raw features makes it inf).
**Blast radius.** All combined-branch scores on any input that deviates from train;
silent because nothing checks finiteness.
**Fix.** Floor the std (`jp.maximum(std, std_floor)` with configurable floor), warn
on degenerate dims, add finiteness guards on scores.
**Proof.** Unit test: constant-latent train batch + off-manifold test point ⇒ error
stays finite and bounded; before fix it is ≥1e6.

### F-04 · Confidence calibration is degenerate and sign-inverted
**Where.** `ars/models/m2__detector/confidence.py:33-37` (MAD with no floor),
`:40-60` (`fit_weights`: unguarded BFGS on tiny VAL, no weight sanity check, no
fallback), `:111-115` (z = (e−median)/(1.4826·MAD+1e-8)).
**Why wrong.** `sem_mad = 0` (F-01) ⇒ `z_sem = e_sem × 1e8`. With 6 val points the
weighted logistic fit returned `w_epi=+217, w_sem=−89, w_comb=−209, bias=+44.5` —
**higher combined reconstruction error ⇒ lower p_anomaly**. Baseline ECE = Brier =
0.80 (anti-calibrated). Nothing validates the fit (no convergence check, no held-out
sanity, no monotonicity requirement).
**Blast radius.** `p_anomaly` and `confidence` shipped to the product; the s3
classifier consumes these as features (L1), spreading the corruption.
**Fix.** MAD floors; require minimum val size; constrain w_comb ≥ 0 (or fit on
scores with a monotone link); fall back to a pure combined-error sigmoid when the
fit is degenerate; add ECE/reliability to the training report.
**Proof.** Calibration test: p_anomaly monotone non-decreasing in e_comb (fails
before: w_comb<0); ECE on synthetic val below a pinned bound.

### F-05 · Robust scaling explodes on quantile-degenerate features
**Where.** `ars/stages/s1__data.py:819-823` (winsorization applied only to the
*fit* stats, never to the transformed data), `:825-839` (IQR guard only when
`|IQR| < eps_normalization = 1e-6`).
**Why wrong.** Features whose IQR is tiny but ≥1e-6 (real corpus:
`avg_word_length_sem_max` IQR=0.0044) produce normalized values up to |42|; squared,
they dominate reconstruction error (baseline EPI MSE ≈ 20·D-scaled; historically
reported ~1200 "MSE" has the same root cause). The detector's attention is
concentrated on the noisiest quantile-degenerate features rather than behavior.
**Blast radius.** EPI branch quality; threshold placement; every downstream score.
**Fix.** Configurable scale floor (e.g. `max(IQR, floor)`); apply winsorization
bounds at transform time (train stats saved and reapplied at inference); report
per-feature normalized ranges in the data report.
**Proof.** Property test: for any input distribution, post-normalization train
values within winsorize quantiles map to a bounded range; unit test with an
IQR=0.004 feature — before fix |z|>40, after ≤ bound.

### F-06 · s3 crashes end-to-end runs; classifier splits unguarded; `Pad.split` cannot handle 0 rows
**Where.** `ars/stages/s3__classifier.py:144-151,164` (60/20/20 stratified re-split
with no minimum-class-count guard); `ars/stages/s2__detector.py:63-78`
(`Pad.split` on an empty frame → `jp.concatenate` of zero parts → ValueError).
**Why wrong.** Small classes ⇒ empty val/test frames ⇒ crash (this is what killed
the shipped-sample baseline run). Even when non-empty-but-tiny, `Optim.descend`'s
val-loss on an empty split is NaN and it silently returns initial parameters
(`m3__classifier/architecture.py:199-209`, finding L6).
**Blast radius.** End-to-end pipeline availability; classifier quality silently
degraded on small classes.
**Fix.** Min-count guard with explicit typed error or documented merge-small-classes
policy; `Pad.split` returns empty tensors for empty frames; NaN guard in descend.
**Proof.** Integration test: 43-trace fixture through the full pipeline (fails at
head before fix); unit test `Pad.split(df.head(0), …).shape == (0, max_len, dim)`.

### F-07 · Sentinel values flow into features (explicit contract violation)
**Where.** `ars/data/features.py:123-127` (`llm_tokens` uses `llm_total_tokens`
including sentinel −1 for LLM spans with missing usage), `:299-303`
(`tool_compression` writes literal −1 into a real-valued feature for non-tool spans,
then signed-log maps it to −0.693), similarly `llm_temperature`-style fields if ever
added; the data spec (`docs/data_requirements/ТЗ…` §7) states aggregating functions
must **ignore** sentinel-valued spans.
**Why wrong.** Sentinels are *codes*, not measurements. −1 inside means/rolling
quantiles biases every aggregate they touch; the model learns sentinel frequency,
not behavior. Silent because values are plausible.
**Blast radius.** EPI feature quality across all corpora with missing LLM usage
(the 1k sample has none missing; production will).
**Fix.** Feature builders mask sentinels to null before aggregation (nulls are
skipped by polars aggregates), with per-feature null policy.
**Proof.** Unit test: an LLM span with `llm_total_tokens=-1` must not shift
`llm_tokens_mean` vs the same frame without that span (before fix it does).

### F-08 · Split unit is (trace_id, agent_id), not trace_id; anomaly-group order unspecified
**Where.** `ars/stages/s1__data.py:600-602` (traces = `group_by('trace_id','agent_id')`),
`:723-748` (normal split + per-class anomaly split; `anom_all.group_by(strata_col)`
iteration feeds `pl.concat` in engine-dependent order).
**Why wrong.** A multi-agent trace becomes several rows; the same `trace_id` can
land in train and test. Trace-level aggregate features (`action_entropy`,
`unique_agents`, … computed `.over('trace_id')`) are shared between those rows —
textbook group leakage. Group iteration order is not a stated polars contract, so
val/test row order can vary across versions/engines.
**Blast radius.** Zero on current single-agent corpora; real production MAS data is
multi-agent by definition — this silently invalidates the split there.
**Fix.** Split strictly by `trace_id` (all agent-rows of a trace travel together);
sort anomaly groups by class name before concat.
**Proof.** Anti-leakage test: no `trace_id` appears in more than one split
(fails before fix on a multi-agent fixture).

### F-09 · `max_len` fitted on train+val+test; silent truncation at inference
**Where.** `ars/stages/s2__detector.py:206` (`Pad.max_len` over all three splits);
`:63-70` (`filter(pl.col('_step') < max_len)` silently drops steps beyond
`max_len`); inference uses the stored `max_len` (`:697-698`).
**Why wrong.** Tensor sizing depends on test data (leak of test statistics into
the training configuration); at serving time longer traces are silently truncated —
precisely the anomalously-long traces the detector should flag.
**Blast radius.** Real-time path correctness for long traces.
**Fix.** `max_len` from train only (configurable cap); explicit truncation policy
with a logged/flagged indicator feature.
**Proof.** Test: a 2×max_len trace at inference either raises per policy or is
flagged; before fix it is silently cut and scores as normal-length.

### F-10 · Embedding model unpinned
**Where.** `ars/configuration/c1__data.py:33` (`embedder_path` points at whatever
directory/zip is provided; no model name/revision/hash recorded);
`S1Meta.embedding_model` stores only the path (`s1__data.py:927`).
**Why wrong.** Different embedder ⇒ different sem vectors ⇒ every checkpoint,
threshold, calibration silently invalid. Nothing detects the swap.
**Blast radius.** Model lifecycle management; all persisted artifacts.
**Fix.** Record model name + revision + weights hash in S1Meta and into every
model artifact; verify at inference load; fail loudly on mismatch.
**Proof.** Test: loading a checkpoint against a different embedder hash raises.

### F-11 · Feature selection is fit on the full pre-split dataset
**Where.** `ars/stages/s1__data.py:398` (`select_features(spans_enriched, …)` before
injection/split — fill-rate, static-rate, correlation pruning see everything),
`:413-421` (`max_abs_threshold = 1e6` filter, also full-data, also hardcoded).
**Why wrong.** Selection statistics use future VAL/TEST rows (selection bias: mild
optimism; correlation pruning can keep/drop different features than train-only
statistics would). The 1e6 filter runs *before* normalization on filled data.
**Blast radius.** Feature-set composition; subtle optimism in all metrics.
**Fix.** Fit selection on train-normal spans only; persist the selected list (it is
already persisted via `S1Meta.epi_features`); make `max_abs_threshold` config.
**Proof.** Anti-leakage test: selection function receives only train rows (fails
by construction before).

### F-12 · Classifier trains on the detector's VAL∪TEST with label-shaped features
**Where.** `ars/stages/s3__classifier.py:137-141` (pool = s1 train+val+test filtered
to anomalies; train parquet contributes zero rows since anomalies live only in
val/test — s1__data.py:750-752), features at `:83-98` include `p_anomaly`,
`confidence` (fit on val labels), `e_comb − best_threshold` (threshold fit on val),
latents from a checkpoint chosen on test (F-02).
**Why wrong.** The classifier's train/val/test all consist of rows whose labels
already shaped its inputs; its "test" metrics are not a holdout of the s2→s3 chain.
**Blast radius.** All classifier metrics; product-facing anomaly-type output.
**Fix (staged).** Short term: document the bias, evaluate the chain on data unseen
by s2 threshold/calibration. Long term: nested split or fit s3 on features that do
not embed val-fitted parameters.
**Proof.** Chain-level evaluation on a fresh injected corpus (numbers drop vs the
self-reported ones — record the delta honestly).

### F-36 · Injector text-noise crashes on ~5% of spans (u64→i64 hash wraparound)
**Where.** `ars/data/anomalies_injection.py:71-108` (`TextNoise.corrupt_col`:
`pos = (idx.hash(seed) % span).cast(pl.Int64)` — `Expr.hash` returns u64; the
modulo result wraps negative under signed promotion; `str.slice` requires u64 and
raises `InvalidOperationError` on negative offsets/lengths).
**Why wrong.** Any corpus with more than a handful of hallucination-victim spans
deterministically crashes stage s1 (observed: 12/252 values negative on the 20k-span
synthetic corpus). The shipped 43-trace sample survives by luck.
**Blast radius.** Training on any realistic corpus is impossible; there is no
obtainable baseline beyond the tiny sample (AUDIT_00 §4.3).
**Fix.** Compute positions with explicit u64 arithmetic or
`(hash % 2**63).cast(Int64)`; property-test the expression over thousands of rows.
**Proof.** Unit test: corrupt_col over 10k synthetic texts never produces negative
slice args (crashes before fix); integration: synthetic corpus run completes s1.

---

## P1 — blocks reliability or scaling

### F-20 · Committed hyperparameters are demo debris
`ars/configuration/experiments/e2__detector.py:41` `epochs: int = 10 #500`,
`patience: 50` (unreachable at 10 epochs), `_train_loop` hardcodes
`max_epochs=10000` when target_loss set (`train.py:175`). Threshold metric for the
second experiment is `precision` — not a choice, an artifact of `build_grid`
cycling `('youden','precision','recall','f1')` by index (`e2__detector.py:188-190,
224-225`). Fix: config-driven epochs/patience/metrics; grid explicit. Proof: config
round-trip test + the Phase 7 run at real epochs.

### F-21 · No NaN/Inf/finiteness guards anywhere in train or serve
No gradient-finiteness checks, no loss NaN aborts, no score finiteness gates
(`train.py`, `s2__detector.py`, `confidence.py`). Combined with F-03/F-05 this is
how a silent 3.9e8 error ships. Fix: optax finite-guard wrapper + score gates +
typed errors. Proof: numerical test — adversarial inputs (all-sentinel, huge
durations) produce finite scores or a typed exception.

### F-22 · Threshold selected at injected prevalence (~20%) is invalid at production prevalence
`stratified_split` puts ~70% of injected anomalies into VAL (`c1__data.py:63`),
giving VAL ≈ 61% anomalous on the real baseline (6 traces). Youden is
prevalence-invariant in expectation but the *operating point* (precision/FPR at the
chosen threshold) is not; at 1% real prevalence the same threshold's PPV collapses
(baseline: precision 0.8 at 50% prevalence ⇒ ≈3.9% at 1% prevalence, holding
recall 0.8 / FPR 0.2). Fix: evaluation must report recall@FPR≤10% and
prevalence-adjusted PPV; threshold selection metric and target prevalence become
config. Proof: eval report contains these; a test pins the PPV formula.

### F-23 · Branch losses are D-scaled and cross-branch incomparable
`ars/models/metrics.py:208-212` divides by timestep count, not element count
(verified: masked loss of all-ones error on D=3 is 3.0). EPI "MSE" is ×45, SEM
×1024, while combined errors are per-element means (`architecture.py:528-532`).
Reported "MSE ~1200" magnitudes are artifacts of this convention. Fix: per-element
normalization everywhere (documented); recalibrate expectations in reports.
Proof: unit test pins masked loss of unit error to 1.0 for any D.

### F-24 · Evaluation surface too thin for the product targets
No per-anomaly-type metrics, no ROC/PR-AUC in the standard path (only in the ad-hoc
`run_inference`), no calibration metrics, no latency measurement in the pipeline.
Fix: evaluation module (Phase 5) emitting all of AUDIT_00 §4's metrics per run.
Proof: golden run report contains all fields.

### F-25 · Determinism gaps around the verified core
Verified bit-identical same-seed runs (AUDIT_00 §6) but: `run_id` unseeded
(`main.py:157`), run dirs wall-clocked (`:158`), `random.choices` before seeding,
polars group-iteration order unpinned (F-08), `Trainer` reuses the same key for init
and first permutation (`train.py:222-232` — rng passed to both `make_train_state`
and `_train_loop`), all experiments share `PRNGKey(cfg.seed)` streams
(`s2__detector.py:376`), s3 seed never varies training independently of split (L11).
Fix: seed-derived run_id, split keys per purpose (`fold_in` discipline).
Proof: determinism suite incl. restart-across-process test.

### F-26 · Error handling erases causes
`ars/main.py:154-162` bare `except:` → `exit(1)` with no cause (also catches the
"same-second run dir" collision `exist_ok=False`); `Embedder.maybe` swallows every
exception and silently degrades semantic directions to the e1 axis
(`anomalies_injection.py:846-853`); `validation.py:861-862` blanket except turns
bugs into red data-quality verdicts; `FileIO` adds no context; pickles carry no
schema/version. Fix: typed exceptions, no silent fallbacks (log + fail or log +
explicit degraded mode), versioned artifacts. Proof: failure-mode tests per case.

### F-27 · Inference path does double work and discards the audit trail
`detect_anomalies` (`s2__detector.py:694-715`) runs `Predict.batch` (which encodes
both branches) then encodes both branches *again* for z-latents; it then filters to
`detector_is_anomaly == True`, so scores for normal traces are dropped — no audit
trail, and s3/s4 can never inspect near-threshold traces. Fix: single encode pass;
emit all traces with scores + flag. Proof: perf measurement (≈2× encode cost
removed); integration test asserts scored output for every input trace.

### F-28 · Observability: prints, colors, monkey-patching; no run manifest
TUI `sprint` everywhere; `perf.benchmark` stores timings *on the function object*
(`perf.py:84-98`, not reentrant); `inject_color_scheme` monkey-patches module
globals (`perf.py:105-107`, racing when two stages patch shared helpers);
`redirect_native_stderr` dup2's fd 2 process-wide (`tui.py:405-426`, raises if fd 2
closed). No structured logs, no single manifest tying inputs→artifacts→metrics.
Fix: `logging` + JSONL run manifest; keep TUI as optional veneer. Proof: a run
produces a manifest with config hash, data hash, seeds, metrics, artifact paths.

### F-29 · Config sprawl and hardcoded environment
String-prefixed kwarg casting duplicated against dataclass types
(`main.py:124-139`); `/tmp/mas-monitor#…`, `/mnt/data/...` defaults sprinkled
(`main.py:158,250-262`, `c1__data.py:33`, `synthesis.py:521`, `validation.py:872`,
injector `:934`); empty config files (`c0__paths.txt`, `c4__rca.py`,
`c5__reports.py`). Fix: single typed config with file+CLI override, explicit paths.
Proof: config round-trip test; grep gate for `/mnt/` literals in CI.

### F-30 · Reports can kill training: altair 5000-row limit + Cyrillic fonts headless
`viz.save_error_distribution` builds a row per sample and altair's default
`max_rows=5000` raises `MaxRowsError` (nothing disables it) — a >5k-span test set
crashes s2 *after* training, before artifacts complete (`viz.py:472-484`,
`s2__detector.py:461-463`). vl-convert with no system fonts renders tofu or errors;
no try/except around chart saving (the good pattern exists only in
`validation.py:743-765`). Fix: guard + disable max_rows + degrade gracefully.
Proof: failure-mode test with 6k-row error frame.

### F-31 · Deployment tree is unshippable as committed
`deploy/deploy.py` is 0 bytes; the per-worker source payloads
(`deploy/nodes/_codebase/data/…`, incl. the `run_all.py` the platform executes) are
absent; `_codebase/main.py:22` prints the entire base64 archive into node logs.
Fix (scoped): generate worker trees from `ars/` at build time; document. Proof:
packaging script builds all five worker payloads from a clean checkout.

### F-32 · Classifier serve/train mismatch and silent label fallback
Trained on true-labeled anomalies; served on detector positives (incl. FPs) with
forced argmax — no reject option (`s3__classifier.py:344-347`); unknown labels
silently map to class 0 (`:117-119`). Fix: 'benign/unknown' class or confidence
gate; strict label vocabulary. Proof: serve-time test with a detector FP expects
non-forced or gated output.

### F-33 · Stacking CV leaks
OOF folds early-stop on the *same* shared val set (`stacking.py:36`); histogram
bin edges fit on the full matrix incl. held-out folds
(`m3__classifier/architecture.py:429-430,460-461`); folds unstratified
(`stacking.py:25-26`); val used for early-stop *and* reported (L3).
Fix: per-fold early-stop sets, bins from fold-train only, stratified folds.
Proof: leakage test — OOF predictions must not change when a held-out fold's
feature values are permuted (they do before the bins fix).

### F-34 · Pipeline has no input validation gate
`load_spans` (`s1__data.py:101-211`) ingests any parquet blind — no schema check,
no per-trace rejection — while a complete spec-driven engine exists unused
(`ars/data/validation.py`; the data spec mandates whole-trace rejection on invalid
mandatory fields). Fix: validation gate (configurable strictness) in the load path,
reusing `Quality.tagged`/spec validities. Proof: contract tests — malformed fixture
traces are rejected with actionable errors; valid ones pass unchanged.

### F-35 · Latency: no batch-1 path budget, JIT warmup unmanaged
Baseline p50 = 40.6 ms/trace CPU (target 20–30 ms GPU; p99 92 ms includes
re-jit noise). Single-trace inference pays `Pad.split` + full `Predict.batch`
overhead; nothing caches compiled functions across calls in a service context; no
measurement exists in-repo. Fix: measured inference module with warmup, fixed
shapes, and a latency report; GPU/CPU device config already exists. Proof:
performance test with budget assertion (CPU budget documented separately).

---

## P2 — maintainability

### F-50 · `duration_diff` is identically zero
`ars/data/features.py:107-111`: `(end−start) − (end−start)`. Likely intended
`duration.diff()` over the sequence. Fix in Phase 5 (behavior change ⇒ after
safety net; adds a real feature). Proof: unit test — nonzero for varying durations.

### F-51 · Dead code inventory
Three large commented-out function bodies in `s1__data.py` (:34-67, :447-502,
:544-566); unreachable real embedding code below the stub; `viz.py` dead helpers
(`save_missingness/reliability/pr/distribution/mapping_bars/training`,
`Frames.to_long`, `Charts.dashboard`); ~180 unused lines in
`recursion_schemes.py` (ana/hylo/para/apo/histo/futu/zygo/Free/Cofree);
`verification.py` orphan (and its `as_repr` compare uses truncated `str(df)` —
false-positive "reproducible"); `code_format.py` CLI broken for its own read-only
mode (`:234` unpack); `sql/trace_request.py` `fetch` discards built params
(`:363-366`); zero-byte files (`c0__paths.txt`, `c4__rca.py`, `c5__reports.py`,
`utilities/codebase_check.py`, two `todo__*` files). Disposition: move to
`legacy/` or delete-with-archive per rules; keep spec/validation live.

### F-52 · Composition DSL disproportionate to use
`identity >> detect >> classify >> analyze` (`main.py:220`) is the sole use of a
262-line composition framework + 282-line recursion-schemes kit, vendored twice
more under `deploy/nodes/*`. Replace with plain function composition; archive.

### F-53 · s4 RCA is a stub, s5 trivially thin
`s4__rca.py` returns `'-dummy-'`. Product's top R&D priority is RCA — the seams
(per-span attribution surface) do not exist. Phase 5 builds the seams, not RCA.

### F-54 · No manifest, no README, no tests, mixed-language logs
No requirements/pyproject (D-1), no README/CLAUDE.md, zero test files, Russian TUI
strings interleaved with English identifiers (grep-hostile logs).

### F-55 · Stage coupling
s3 imports `Pad`/`load_models_for_inference` from the s2 *stage* module
(`s3__classifier.py:18`) — stages not independently importable; `Pad` belongs in a
shared module.

### F-56 · Sample/spec dtype drift unhandled by default
`meta_langgraph_step` Float64 vs spec Int64; Categorical vs Enum. `recast=False` is
the default (`c1__data.py:70`) — the drift silently propagates. Fix: validation
gate (F-34) handles; recast default documented.

---

## P3

### F-70 · TUI truecolor + `blink` escapes garble CI logs (`tui_data.py:34,57,71,80`); progress-hook regex matches only the English epoch line (`tui.py:376-377`).
### F-71 · `viz` writes `.json` before `.png` — stray artifacts on render crash (`viz.py:482`).
### F-72 · `validation.py:759` `mkdtemp()` leaks temp dirs; `rejection()` reports span-weighted rates labeled as trace fractions (`validation.py:226-235` vs report note).
### F-73 · `Trainer._run_lstm_epoch` drops the last partial batch each epoch (`train.py:114-118`) — acceptable, but undocumented.
### F-74 · Bi-LSTM encoder takes the merged output at the last valid step, where the backward half has seen exactly one token (`architecture.py:221-223`) — architectural choice worth revisiting, not a bug.

---

## Coverage of the mandated audit checklist

| Checklist item | Verdict | Finding |
|---|---|---|
| Normalization fit only on train normals | **Yes** (correct) — `normalize_epi_features` filters `is_anomaly==0` from a train set that is all-normal by construction | — |
| Anomalies excluded from train | **Yes** (correct) — `train = norm_train` (`s1__data.py:750`) | — |
| Threshold on VAL, evaluated on TEST | Threshold: yes. But experiment selection on TEST | F-02 |
| Split grouped by trace | **No** — grouped by (trace, agent) | F-08 |
| Stratification by subtype preserved | Yes for anomalies (per-class val/test split); normals unstratified (fine) | — |
| Padding/masking in loss | **Correct** — masked loss excludes pads; but D-scaled convention | F-23 |
| Bi-LSTM sequence lengths | Correct (`nn.RNN(seq_lengths)`); encoder tap point noted | F-74 |
| Decoder train/infer mismatch | **None** — autoregressive decoder feeds back its own predictions in both (no teacher forcing at all; slow but consistent) | — |
| Sentinel poisoning | **Yes, real** | F-07 |
| BatchNorm @ batch 8 | Present (combined AE, `use_batch_norm=True` default); batch_stats persisted and used at eval; risk noted, superseded in practice by F-03/F-04 magnitudes | F-22 note |
| Early stopping | Monitors val-normal loss; **restores best** state (correct); but epochs=10 makes it moot | F-20 |
| Threshold vs production prevalence | **Not held** — quantified | F-22 |
| Seeds/PRNG | Core deterministic (verified); discipline gaps | F-25 |
| Numerics | Multiple: overflow, no guards, D-scaling, robust-scale explosion | F-03/F-05/F-21/F-23 |
| Embedding pinning | **Unpinned** | F-10 |
| Loader enforces schema | **No** | F-34 |
| Trace rejection per spec | Implemented in unused module only | F-34 |
| session_id flags handled | Validated in spec expr (`spec.py:571-591`), never consumed as features or checks in pipeline | F-34 note |
| Single source of truth for schema | **Yes** — `ars/specification/spec.py` is genuinely good and reused by validation/synthesis; features hardcode column names as strings but they resolve to the same spec names | Partial credit |
