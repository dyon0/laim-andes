# Deploying to SberDS

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
  and writes a **self-contained model bundle** (zip) to `model_store_dir`
  (default `/mnt/data/laim/models`); the path is emitted on `model_out`.
* **inference**: `model_in` (the bundle, via port or `model_path` param) +
  `path_traces_infer` → scores every trace (full audit trail + RCA
  attribution) and emits the product contract (`anomaly_traces` dataframe,
  `test_anomalies` JSON — field-compatible with the legacy end2end node).

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
