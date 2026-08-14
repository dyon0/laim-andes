# Deploying to SberDS

**Status: WORKING end-to-end** — confirmed by the operator on 2026-08-14:
train (real corpus, injected anomalies, s3 classifier) → model bundle over
the `model_out`→`model_in` port wire → inference with the full product
contract, on the production 8×H100 node. The findings log below records
every platform quirk that had to be handled to get here.

## Current model: one dual-mode node (this repo IS the node)

The platform builds a node directly from this git repository:
**`descriptor.json`** (repo root) + **`run.py::main(**params)`** (platform entry)
+ `requirements.txt` (auto-installed into the `py312-gpu` base image).
No codebase-smuggling nodes are needed anymore — see "Legacy descriptors" below.

One node type, two modes (`mode` UI parameter):

```
             ┌─────────────────────────────┐          ┌──────────────────────────────┐
 spans ────▶ │ laim-detector  mode=train   │          │ laim-detector  mode=inference │
 embedder ─▶ │                             │          │                              │
             │ model_out ─── path to zip ──┼────────▶ │ model_in                     │
             │ eval_report / html_reports  │          │ path_traces_infer ◀── spans  │
             │ detector_metrics_holdout    │          │ path_embedder    ◀── embedder│
             └─────────────────────────────┘          │ anomaly_traces / test_anomalies
                                                      └──────────────────────────────┘
```

* **train**: `path_traces_train` (+ optional `path_traces_infer` to score
  immediately) → trains s1→s2 (+s3 when the corpus supports it), evaluates,
  and packs a **self-contained model bundle** (zip). `model_out` carries the
  bundle **bytes** base64-inline in its JSON payload (+sha256) — the only
  thing a port wire actually transfers is the payload, and the zip on disk
  dies with the train container (proven 2026-08-14 08:51: model_in received
  75 bytes of path string). No shared storage is needed for the hand-off;
  `model_store_dir` matters only for bundles > 256 MB or the `model_path`
  pattern.
* **inference**: `model_in` (the payload from `model_out`, or `model_path`
  to a zip/dir on storage) + `path_traces_infer` → verifies the sha256,
  scores every trace (full audit trail + RCA attribution) and emits the
  product contract (`anomaly_traces` dataframe, `test_anomalies` JSON —
  field-compatible with the legacy end2end node).

Ports unused by the selected mode are declared `required: false` and can stay
unconnected. Instantiate the same node twice in a project to build the
train→inference pipeline shown above; inference instances can also run on a
T+1 schedule against a fixed `model_path`.

### Why one node instead of a split

Each SberDS node is built from its own git repository. Splitting
train/inference (or synth/validate) into separate nodes would mean N clones of
this repo to keep in lockstep — version-skew risk with no benefit, since the
code already separates training from serving internally (`laim/pipeline.py`)
and the bundle is the only hand-off. Auxiliary utilities (synthetic data,
standalone contract validation) remain available via `run.py synth|validate`
outside the platform.

### The model bundle

`laim_model_<run_id>.zip` contains the trained model pickles, `s1_meta.json` /
`s2_meta.json` (+ `s3_meta.json` when the classifier trained), the run
manifest and the eval report. Internal paths are stored bundle-relative and
absolutized on extraction, so a bundle trained on a GPU host serves anywhere.
Safety at load time: the bundle must contain `s2_meta.json` (typed error
otherwise), the **embedder fingerprint** recorded at training is verified
against `path_embedder` (serving with a different embedding model raises,
finding F-10), and the training-time embedding parameters are re-applied from
the bundle's manifest so inference vectors match training vectors.

### GPU / CPU

`script.baseImageKey` is `py312-gpu`; the actual device is the `device` UI
select (`cpu`/`gpu`), applied to the JAX detector and the embedder alike.
CPU works everywhere (validated); GPU is the production target for the real
`deepvk/USER-bge-m3` embedder.

### Multi-GPU

Embedding — 80–95 % of the GPU wall time on large corpora — is
**data-parallel across all visible GPUs** in `device = gpu` mode: one model
**replica per GPU driven by threads** (ordered contiguous slices, gather in
order — vectors are identical to single-device encoding up to float
summation order, verified by test). Threads, never processes: the SberDS
pywrapper is **not spawn-safe** — a spawned worker re-executes the whole
node (proven in the 2026-08-13 23:49 run), and fork is unusable once CUDA
is initialized. Controls:

| Parameter | Default | Meaning |
|---|---|---|
| `embedding_gpus` | `0` | `0` = all visible GPUs, `N` = first N, `1` = single GPU |
| `embedding_pool_chunk` | `5000` | texts handed to each replica per dispatch |

Throughput tuning (adversarial review, unbenchmarked here — no GPU in the
dev container): each dispatch is a sync barrier (all replicas wait for the
slowest slice), slices are split by text COUNT not token length, and the
default `embedding_batch_size = 32` is small for an H100 at
`embedding_max_length = 1024`. If 8-GPU embedding throughput matters on a
real corpus, benchmark `embedding_batch_size` 128–256 first — it is the
single biggest lever — and expect somewhat sub-linear scaling versus 8×.

The s2 detector autoencoders are small and train on **one** GPU by design;
parallelizing the experiment grid across GPUs is recorded as future work in
PLAN.md. So on an 8×H100 host expect: all 8 busy during embedding, one busy
during training — that is the intended resource profile, not a bug.

Memory policy: the node sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` (an
operator value in the environment wins) because JAX's legacy default —
preallocating 80 % of EVERY visible device — would starve the encoding
workers. The full GPU inventory, the memory policy, and the chosen worker
set are logged at node start and recorded in the run manifest
(`metrics.gpu_topology`).

### What the platform actually looks like (probe, 2026-08-13)

Measured by `deploy/probe_node/` on the production `py312-gpu` image — these
facts drive the packaging and the sizing advice below:

| Fact | Value | Consequence |
|---|---|---|
| Preinstalled torch | `2.8.0+cu128`, works | `requirements.txt` must never pin torch (the mirror's `+xpu` builds outrank `+cu128` — that caused the `libsycl.so.9` crash) |
| Driver / CUDA | 570.86.15 / **12.8 ceiling** | only cu12 wheels can run; JAX uses `jax-cuda12-plugin==0.11.0` |
| GPUs | 8 × H100 80GB, `CUDA_VISIBLE_DEVICES` unset | embedding (the dominant GPU cost) runs data-parallel on all of them (`embedding_gpus`, below); the small s2 autoencoders train on one GPU |
| CPU quota | cgroup `quota/period` (probe run: 8 cores; host shows 128) | `run_node` sets `POLARS_MAX_THREADS`/`OMP_NUM_THREADS` from the quota — otherwise polars spawns 128 threads into an 8-core cap |
| Memory limit | 453.5 GB (cgroup) | see sizing below |
| Disk | 1.5 TB free on `/tmp` and `/opt/module` | port staging of 2×56 GB is fine |
| `/mnt/data` | **permission denied** | bundle store falls back to `/tmp` with a warning; cross-node hand-off needs an admin-provided shared path (OQ-6) |
| Dataframe ports | stored on HDFS as parquet parts with **positional column names** (`"0".."57"`); the real schema lives in port metadata and is applied only when the PLATFORM parses the port. So: `getPortAsLocalPath` returns nameless, unusable parts (run 22:16); in-memory delivery has correct names but Booleans cast to strings (run 21:52) | data in-ports are `"dataframe"` **without** `getPortAsLocalPath`; the in-memory payload is repaired via the spec `Recast.overlay` and staged back to parquet. A schema preflight rejects nameless inputs with an actionable hint. Directory-of-parts paths still work for locally-produced data (`spans_scan_source`) |
| Model ports | extension-less blob (`unstructured_data`), ZIP by magic bytes | `_resolve_embedder` sniffs content, never suffixes |
| pip index | only `sberosc.ca.sbrf.ru` reachable (PyPI mirror + sber-pytorch incl. `+xpu`) | pins must resolve there; they are plain-PyPI packages |
| Absent from image | jax, flax, optax, polars, sklearn, sentence-transformers, transformers, altair | installed by `requirements.txt` on node build |
| torchaudio in image | `2.8.0+xpu` — an Intel build paired with CUDA torch, **broken** (`libtorch_xpu.so` missing; crashed run 2026-08-13 20:04 inside `import transformers`) | quarantined at node start (`sys.modules` block marker → transformers runs without audio support, which this node never uses); worth reporting to the platform team — it breaks any tenant that imports torchaudio |

### Sizing (CPU / RAM) for large corpora

Yes — **GPU mode still uses the CPU heavily**: everything in s1 (parquet IO,
polars feature engineering, injection, validation, embedder *tokenization*) is
CPU work; the GPU only runs the embedder forward pass and the JAX
autoencoders. Measured on the reference sample, a snappy spans parquet expands
**~5.1×** in memory (98 % of it text), and each span adds a 4 KB fp32
embedding; with padded sequence tensors the eager pipeline peaks at roughly
**12–25× the on-disk size**.

Practical guidance for the observed limits (453 GB RAM, up to 40 CPUs):

- **CPU: request 32–40 cores.** Below ~8, s1 and tokenization dominate wall
  time; the first run's 1-core default is why embedding "hung".
- **RAM: request the maximum (the 453 GB limit).** That safely covers inputs
  up to ~15–20 GB of snappy parquet with the current in-memory pipeline.
- **A 56 GB corpus does NOT fit the current eager pipeline** (~286 GB raw
  frame + ~212 GB span embeddings + ~340 GB padded tensors ≫ 453 GB). Until
  the out-of-core s1 path lands (PLAN.md "Remaining work"), train on a sampled
  subset (5–20 GB is statistically ample for the normal-behavior autoencoders)
  and score large corpora in chunks. `cmd_prepare` now warns before the OOM
  instead of dying mid-run.

### Sizing the INFERENCE instance (much smaller than train)

Inference skips injection, training and the experiment grid — its cost is
feature engineering + embedding + a forward pass. **CPU-only inference is
viable** for periodic batch scoring, so the inference instance does not have
to hold a GPU:

| Input per run | Device | CPU | RAM | Expected wall time |
|---|---|---|---|---|
| ≤ ~100 MB parquet (≈ 40k spans) | `cpu` | 16 | 16–32 GB | minutes (embedding dominates: with `embedding_max_length=256` typically 1–3 min; at 1024 up to ~10 min) |
| ≤ ~1 GB parquet | `cpu` | 32 | 64 GB | tens of minutes — the practical CPU ceiling |
| larger, or tight schedules | `gpu` (1 GPU is enough — `embedding_gpus=1`) | 8–16 | 25× input size | embedding drops to seconds–minutes |

Notes: the detector forward pass is trivial everywhere (measured 32.7 ms/trace
p50 on CPU at batch 1; batched scoring is far faster) — the embedder is the
whole story, and `embedding_max_length` is the big CPU lever (256 ≈ 4× faster
than 1024 on real text). RAM follows the same ~25× on-disk rule as training
but inputs are far smaller; the model bundle itself is MB-scale.

### Testing the node contract

`tests/test_platform.py` pins the descriptor against the adapter: entry point
is `run.py::main`, out-port names equal the returned dict keys, every UI
parameter is understood, mode-dependent ports are optional, the bundle
round-trips, and (slow test) a real train→bundle→inference hand-off runs end
to end through `run.py::main` exactly as the platform calls it.

## Legacy descriptors (`deploy/descriptors/`, `deploy/nodes/`)

Kept for reference only. They implement the old two-node-per-stage pattern
(`src-*` codebase nodes emitting a base64 tarball + checksum into `env-*`
worker nodes) from the era when the platform accepted only a single `main.py`
per node. With git-repo nodes this is obsolete: the `received_codebase` /
`received_checksum` ports, the packing/unpacking node code and the five
per-stage worker repos are superseded by the single root `descriptor.json`.
Do not register the legacy descriptors alongside the new node.
