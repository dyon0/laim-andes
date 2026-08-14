# OPEN QUESTIONS

Decisions taken autonomously with conservative defaults, per working rules.
Each entry: question / options / chosen default / rationale.

---

## OQ-1: Branch name mismatch

**Question.** The mission brief says `BRANCH: new_version`, but the remote-execution
harness designates `claude/lumimas-anomaly-refactor-7stpda` as the only branch this
session may push to, and forbids pushing anywhere else.

**Options.** (a) create `new_version` locally and push (violates harness constraint);
(b) develop on the designated branch.

**Default taken: (b).** All work lands on `claude/lumimas-anomaly-refactor-7stpda`.
It can be renamed/merged to `new_version` by a human with push rights.

---

## OQ-2: No GPU in this container

**Question.** The brief states 2×H100; this container has 4 CPU cores, 15 GB RAM and
no GPU. How to size baseline/validation training runs?

**Options.** (a) refuse to run anything until GPU appears; (b) run CPU-sized corpora
and honestly record hardware in every metrics artifact.

**Default taken: (b).** The framework must work on CPU anyway (explicit requirement).
All baseline/validation artifacts record `gpu: none (CPU-only container)`. The 20 h
training budget is interpreted as a wall-clock ceiling on the target hardware; CPU
runs here are scaled (smaller corpora / epochs recorded in configs) so nothing runs
unbounded.

---

## OQ-3: Embedding model unobtainable (network policy)

**Question.** `deepvk/USER-bge-m3` must be pinned and cached, but huggingface.co is
blocked by the container network policy (proxy 403). The legacy pipeline needs *a*
SentenceTransformer directory for the anomaly injector (s1 passes a real embedder to
`inject_anomalies`), even though normal-span embeddings are stubbed to zeros
(s1__data.py:505).

**Options.** (a) block; (b) modify code to skip the embedder (changes behavior before
the baseline freeze); (c) build a deterministic random-weight stand-in with the same
interface (SentenceTransformer dir, 1024-dim, L2-normalized, local_files_only).

**Default taken: (c).** Code path unchanged; the stand-in is seeded and rebuildable
(`baseline/build_standin_embedder.py`). Every artifact that used it says so. On the
target infrastructure the real model must be downloaded once and pinned by revision;
this is recorded as a gap (embedding model pinning) in the audit.

---

## OQ-4: Baseline corpus — 43 traces is statistically meaningless

**Question.** The only real data is `data/traces_1k_sample.parquet` = 1000 spans =
43 traces (single agent). The spec itself recommends 1e5–1e6 traces. A 43-trace
baseline yields test sets of a handful of traces.

**Options.** (a) baseline only on the 43 traces; (b) baseline only on synthetic;
(c) both: the real sample (honest but noisy) plus a corpus from the repo's own
deterministic synthesizer (stable numbers, same pipeline).

**Default taken: (c).** Both recorded in `baseline/baseline_metrics.json` with
corpus provenance. The real-sample numbers are labeled as high-variance.

---

## OQ-5: `epochs = 10` in the committed experiment grid

**Question.** `ars/configuration/experiments/e2__detector.py:41` has `epochs: int = 10
#500` — the committed default was cut from 500 (comment left in place), so the
"existing pipeline" trains for 10 epochs with patience 50 (early stopping can never
trigger). Is the baseline the committed 10-epoch config or the commented-out 500?

**Options.** (a) baseline at 10 epochs (what the code actually does); (b) restore 500.

**Default taken: (a).** The baseline freezes what is committed. The 500-epoch
behavior is evaluated later in Phase 7 on the refactored pipeline, where epochs are
config, not code.

---

## OQ-6: No writable shared storage for the model-bundle hand-off (SberDS)

**Question.** Train mode packs the model into a portable zip and returns its PATH
on `model_out`; the inference node reads that path. This requires a directory that
is (a) writable by the train node and (b) readable by the inference node. The
environment probe (2026-08-13) showed `/mnt/data` is permission-denied for the
node's service account, and out-ports carry JSON payloads, not files — so today
there is NO storage the hand-off can use.

**Options.** (a) platform admin grants a writable shared directory (set it as
`model_store_dir`); (b) run train+inference in one node invocation (already
supported: connect `path_traces_infer` in train mode — no hand-off needed);
(c) base64-embed the bundle in the `model_out` JSON payload (works for MB-scale
bundles, untested against platform payload limits).

**Default taken: (b) + fallback.** The node no longer crashes on an unwritable
store: it falls back to /tmp with a loud warning (bundle valid within the run,
lost with the container). (a) is the real fix — ask the platform admin which
path is writable and shared.

**RESOLVED 2026-08-14 — option (c) implemented and it closes the question.**
The first real train→inference wiring proved the failure concretely: model_in
received 75 bytes — the JSON-encoded path string, pointing into the dead
train container (log 08:51). `model_out` now carries the bundle ZIP bytes
base64-inline in its JSON payload (+sha256, verified on receipt); inference
reconstructs the bundle from the port alone. NO shared storage is needed for
the hand-off anymore. `model_store_dir` remains as an optional escape hatch
for bundles over 256 MB (payload embedding is skipped with a warning) and for
the `model_path` pattern; legacy path-only payloads fail with an actionable
message instead of `BadZipFile`.

## OQ-7: Port delivery mode is not uniform (SberDS) — CONFIRMED, both handled

**Confirmed 2026-08-13 21:52 on the main node.** The same data-port
declaration produced a local directory of parquet parts in one run and an
IN-MEMORY pandas DataFrame in the next (crash: pandas ambiguous-truthiness at
`build_config`). Worse, the platform's own parquet→pandas read casts Boolean
columns to strings (`!!! WARNING !!! Column llm_stream is casted from bool to
string(Nominal)` in its log) — silent contract corruption, not just a type
change.

**Resolution (implemented).** `run_node` normalizes all port payloads up
front: data ports arriving in memory are repaired through the spec's
`Recast.overlay` (contract dtypes restored, sentinels honored, unknown
columns untouched) and staged back to parquet; model ports arriving as
anything but a path fail with a clear message. Per the platform team, data
in-ports are now typed `"dataframe"` in the descriptor.

**Third finding (2026-08-13 22:16): raw-file delivery of a dataframe port is
UNUSABLE.** With `getPortAsLocalPath` restored, the node received the parts
verbatim — and they carry POSITIONAL column names (`"0".."57"` + `class`/
`anomaly_type`). The platform stores dataframe ports namelessly and applies
the real schema from port metadata only when parsing the port itself (its
log: `Use columns names from read parameters`). Same upstream, byte-identical
size, two different schemas seen. Consequences:
- `getPortAsLocalPath` is REMOVED from the data in-ports — in-memory delivery
  (+ our dtype repair) is the only route that carries correct names.
- A schema preflight (`ensure_core_spans_columns`) refuses inputs lacking
  `trace_id`/`agent_id` with an actionable hint — before `recast=true` can
  sentinel-fill 46 columns and "train" on one fake trace (which is exactly
  what the gate log showed: "all 1 traces conform" on alien data).
- OPEN at scale: in-memory delivery cost the platform a 7.4 GB pandas parse
  for a 342 MB port — 56 GB corpora will need either schema-preserving file
  delivery from the platform team or reading raw parts + schema from
  `direct_port_links` (the upgraded probe now dumps parquet footers and
  `direct_port_links` to settle this).
