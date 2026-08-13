"""SberDS platform adapter — the node behind `descriptor.json`.

The platform builds this repo into a docker node (deps auto-installed from
requirements.txt), stages input-port files locally (getPortAsLocalPath) and
calls `run.py::main(**params)`. `params` = UI form fields + port values.
The returned dict's keys must match the descriptor's OUT port names.

One dual-mode node:
  mode = "train"      spans → prepare → detector (+classifier) → eval →
                      self-contained model bundle (zip) on shared storage,
                      path emitted on `model_out`; optionally scores
                      `path_traces_infer` in the same run.
  mode = "inference"  model bundle (port file or shared path) + spans →
                      full audit-trail scoring + the legacy product contract
                      (`anomaly_traces` dataframe / `test_anomalies` JSON).

Model bundles are portable: metadata JSONs inside carry paths RELATIVE to the
bundle root and are absolutized on extraction, so a bundle trained on one
machine serves on any other (embedder identity is fingerprint-checked, F-10).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePath
from typing import Any

log = logging.getLogger('laim.platform')


def _effective_cores() -> int | None:
    """CPU budget from the cgroup quota (v2 then v1), None if unlimited.
    On SberDS `os.cpu_count()` reports the host (128) while the container is
    capped by quota (probe: 800000/100000 -> 8 cores)."""
    try:
        raw = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if raw[0] != 'max':
            return max(1, int(int(raw[0]) / int(raw[1])))
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
        period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return None


def _align_thread_env() -> None:
    """Cap thread pools to the real CPU budget BEFORE polars/torch load.

    Without this polars sizes its pool from os.cpu_count() (128 on the
    platform hosts) and oversubscribes an 8-core quota into pure contention.
    Values already set in the environment are respected.
    """
    cores = _effective_cores()
    if cores is None:
        return
    for var in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS'):
        if not os.environ.get(var):
            os.environ[var] = str(cores)
    log.info('thread alignment: cgroup quota = %d cores; POLARS_MAX_THREADS=%s '
             'OMP_NUM_THREADS=%s', cores, os.environ['POLARS_MAX_THREADS'],
             os.environ['OMP_NUM_THREADS'])

# UI form parameter → config path (values reuse laim.config coercion,
# including JSON lists, e.g. experiments=["hub_mse_mse_08_4"]).
PARAM_MAP: dict[str, str] = {
    'device':               'runtime.device',
    'seed':                 'runtime.seed',
    'recast':               'runtime.recast',
    'log_level':            'runtime.log_level',
    'validation_gate':      'data.validation_gate',
    'inject_anomalies':     'data.inject_anomalies',
    'embedding_batch_size': 'data.embedding_batch_size',
    'embedding_max_length': 'data.embedding_max_length',
    'embedding_gpus':       'data.embedding_gpus',
    'embedding_pool_chunk': 'data.embedding_pool_chunk',
    'norm_train_ratio':     'data.norm_train_ratio',
    'norm_val_ratio':       'data.norm_val_ratio',
    'anom_val_ratio':       'data.anom_val_ratio',
    'anom_test_ratio':      'data.anom_test_ratio',
    'epi_normalization':    'data.epi_normalization',
    'winsorize_epi':        'data.winsorize_epi',
    'scale_floor':          'data.scale_floor',
    'norm_z_clip':          'data.norm_z_clip',
    'threshold_metric':     'detector.threshold_metric',
    'select_metric':        'detector.select_metric',
    'select_on':            'detector.select_on',
    'experiments':          'detector.experiments',
    'epochs':               'detector.epochs',
    'patience':             'detector.patience',
    'n_thresholds':         'detector.n_thresholds',
    'encode_chunk':         'detector.encode_chunk',
    'seq_pad_chunk':        'detector.seq_pad_chunk',
    'classifier_enabled':   'classifier.enabled',
    'classifier_seed':      'classifier.seed',
    'target_fpr':           'eval.target_fpr',
    'assumed_prevalence':   'eval.assumed_prevalence',
    'attribution_top_k':    'eval.attribution_top_k',
    'latency_reps':         'eval.latency_reps',
    'output_root':          'paths.output_root',
}

# node-level params handled outside the config tree
NODE_PARAMS = ('mode', 'model_store_dir', 'model_path', 'config_overrides',
               'path_traces_train', 'path_traces_infer', 'path_embedder', 'model_in')

OUT_PORTS = ('model_out', 'detector_metrics_holdout', 'classifier_metrics_holdout',
             'eval_report', 'anomaly_traces', 'test_anomalies', 'html_reports',
             'manifest')

# bundle members whose stored paths are bundle-root-relative
_BUNDLE_META_PATH_KEYS = {
    's2_meta.json': ('output_dir', 'experiment_dir'),
    's3_meta.json': ('output_dir', 'experiment_dir'),
}


# ports that carry spans data (may arrive as a path OR an in-memory frame)
_DATA_PORTS = ('path_traces_train', 'path_traces_infer')
# ports that must arrive as a local file/directory path
_FILE_PORTS = ('path_embedder', 'model_in')


def _stage_dataframe_port(name: str, frame: Any) -> str:
    """Write an in-memory dataframe port payload back to parquet.

    Observed 2026-08-13 21:52: the platform delivered `path_traces_train` as
    a parsed pandas DataFrame instead of a local path — and its own
    parquet->pandas read CASTS Boolean columns to strings (`!!! WARNING !!!
    Column llm_profanity_check is casted from bool to string(Nominal)` in the
    platform log). The payload therefore goes through the spec's Recast
    overlay, which restores every contract column to its contract dtype
    (tolerant boolean parsing, sentinel handling) before anything trains on
    it. Unknown columns pass through untouched.
    """
    import polars as pl
    from ars.specification.spec import Recast

    if isinstance(frame, pl.LazyFrame):
        df = frame.collect()
    elif isinstance(frame, pl.DataFrame):
        df = frame
    else:
        try:
            df = pl.from_pandas(frame)
        except Exception as exc:
            raise ValueError(
                f'порт {name}: неподдерживаемый тип полезной нагрузки '
                f'{type(frame).__module__}.{type(frame).__qualname__} — ожидается '
                f'путь к parquet или pandas/polars DataFrame') from exc

    before = dict(df.schema)
    repaired = Recast.overlay(df.lazy(), df.schema).collect()
    changed = {c: (str(before[c]), str(t)) for c, t in repaired.schema.items()
               if c in before and before[c] != t}
    out = Path(tempfile.mkdtemp(prefix=f'{name}_')) / f'{name}.parquet'
    repaired.write_parquet(out)
    log.info('port %s arrived as in-memory %s (%d rows, %d cols) — staged to %s; '
             'dtypes repaired to contract: %s',
             name, type(frame).__qualname__, repaired.height, repaired.width, out,
             changed or 'none needed')
    return str(out)


def _normalize_port_params(params: dict[str, Any]) -> dict[str, Any]:
    """Both port delivery modes must work regardless of descriptor typing:
    the same declaration produced a local path in one run and an in-memory
    DataFrame in the next (OQ-7). Data ports are staged back to parquet;
    model ports have no meaningful in-memory form and fail with a clear
    message instead of pandas' ambiguous-truthiness error."""
    out = dict(params)
    for name in _DATA_PORTS:
        value = out.get(name)
        if value is not None and not isinstance(value, (str, os.PathLike)):
            out[name] = _stage_dataframe_port(name, value)
    for name in _FILE_PORTS:
        value = out.get(name)
        if value is not None and not isinstance(value, (str, os.PathLike)):
            raise ValueError(
                f'порт {name} должен приходить локальным путём (getPortAsLocalPath), '
                f'получен {type(value).__module__}.{type(value).__qualname__} — '
                f'проверьте тип порта в descriptor.json')
    return out


def _backend_mismatch(torchaudio_version: str, torch_version: str) -> bool:
    """True when exactly one of the two is an Intel XPU build (local version
    tag `+xpu`) — such a torchaudio dlopens libtorch_xpu.so, which a CUDA
    torch does not ship."""
    return ('xpu' in torchaudio_version) != ('xpu' in torch_version)


def _quarantine_broken_torchaudio() -> None:
    """The py312-gpu image pairs torchaudio 2.8.0+xpu with torch 2.8.0+cu128
    (run log 2026-08-13 20:04: OSError libtorch_xpu.so inside
    `import transformers`). transformers' availability guard checks only that
    the package is INSTALLED (find_spec), not that it imports, so the broken
    build explodes at `import torchaudio` in its audio utils.

    We never process audio. Marking the module as blocked via the documented
    `sys.modules[name] = None` convention makes find_spec return None, so
    transformers takes its normal no-torchaudio path instead of crashing.
    Consistent pairings (both CUDA or both XPU) are left untouched.
    """
    import importlib.metadata as md
    try:
        ta, th = md.version('torchaudio'), md.version('torch')
    except md.PackageNotFoundError:
        return
    if _backend_mismatch(ta, th) and 'torchaudio' not in sys.modules:
        sys.modules['torchaudio'] = None
        log.warning(
            'torchaudio %s is built for a different backend than torch %s '
            '(broken pairing in the platform image) — torchaudio quarantined; '
            'transformers runs without audio support (unused by this node)',
            ta, th)


def _resolve_embedder(raw: str | None) -> str | None:
    """Port may deliver a model directory or an archive blob.

    The platform hands model ports over as an EXTENSION-LESS file (observed
    name: `unstructured_data`, ZIP by magic bytes), so the format is sniffed
    by content, never by suffix. Archives are extracted once per run.
    """
    if not raw:
        return None
    p = Path(raw)
    if p.is_dir() or not p.exists():
        return str(p)   # model dir, or let the embedder loader report a miss
    target = Path(tempfile.mkdtemp(prefix='embedder_'))
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as zf:
            zf.extractall(target)
    elif tarfile.is_tarfile(p):
        with tarfile.open(p) as tf:
            tf.extractall(target, filter='data')
    else:
        with open(p, 'rb') as f:
            head = f.read(8)
        raise ValueError(
            f'порт path_embedder: файл {p} не распознан (первые байты: '
            f'{head.hex()}). Ожидается каталог модели sentence-transformers '
            f'или zip/tar-архив с ним.')
    # the model root is wherever config.json lives (archive root or one level in)
    if (target / 'config.json').exists():
        return str(target)
    hits = sorted(target.rglob('config.json'))
    if hits:
        return str(hits[0].parent)
    entries = [e for e in target.iterdir() if e.is_dir()]
    return str(entries[0] if len(entries) == 1 else target)


def build_config(params: dict[str, Any]):
    from laim.config import load_config
    overrides = [f'{PARAM_MAP[k]}={params[k]}'
                 for k in PARAM_MAP if params.get(k) not in (None, '')]
    extra = str(params.get('config_overrides') or '')
    overrides += [line.strip() for line in extra.replace(';', '\n').splitlines()
                  if line.strip() and not line.strip().startswith('#')]
    cfg = load_config(None, overrides)

    # dataframe ports arrive as a DIRECTORY of part files — normalize once here
    from laim.config import spans_scan_source
    port_overrides = []
    if params.get('path_traces_train'):
        port_overrides.append(
            f"paths.train_spans={spans_scan_source(params['path_traces_train'])}")
    if params.get('path_traces_infer'):
        port_overrides.append(
            f"paths.infer_spans={spans_scan_source(params['path_traces_infer'])}")
    embedder = _resolve_embedder(params.get('path_embedder'))
    if embedder:
        port_overrides.append(f'paths.embedder={embedder}')
    return load_config(None, overrides + port_overrides)


def _writable_store(store: Path) -> Path:
    """Verify the bundle store is writable BEFORE training results depend on it.

    Probe (2026-08-13): `/mnt/data` is permission-denied for the node's service
    account. Falling back to /tmp keeps train mode alive (bundle path is still
    returned on `model_out` and inference-in-the-same-run still works), but the
    bundle will NOT survive the container — cross-node hand-off needs a shared
    writable directory (OPEN_QUESTIONS OQ-6).
    """
    try:
        store.mkdir(parents=True, exist_ok=True)
        probe = store / '.laim_write_probe'
        probe.write_bytes(b'')
        probe.unlink()
        return store
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / 'laim' / 'models'
        fallback.mkdir(parents=True, exist_ok=True)
        log.warning(
            'model_store_dir %s is not writable (%s) — falling back to %s. '
            'The bundle will NOT outlive this container: set model_store_dir '
            'to a shared writable path for train->inference hand-off (OQ-6).',
            store, exc, fallback)
        return fallback


# ---------------------------------------------------------------- bundling

def create_bundle(run_dir: Path, store_dir: Path) -> Path:
    """Pack a finished training run into a portable zip.

    Contents: meta/manifest/eval JSONs (paths rewritten bundle-relative) plus
    the model artifact tree. Returns the zip path on `store_dir`.
    """
    run_dir = Path(run_dir)
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    s2_raw = json.loads((run_dir / 's2_meta.json').read_text())
    models_root = Path(s2_raw['output_dir'])

    members: dict[str, Path] = {}
    for name in ('s1_meta.json', 's2_meta.json', 's3_meta.json',
                 'manifest.json', 'eval_report.json'):
        if (run_dir / name).exists():
            members[name] = run_dir / name
    for src in models_root.rglob('*'):
        if src.is_file() and src.suffix in ('.pkl', '.json'):
            members[f'models/{src.relative_to(models_root).as_posix()}'] = src

    bundle_path = store_dir / f'laim_model_{run_dir.name}.zip'
    with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for arcname, src in members.items():
            if arcname in _BUNDLE_META_PATH_KEYS:
                meta = json.loads(src.read_text())
                for key in _BUNDLE_META_PATH_KEYS[arcname]:
                    absolute = Path(meta[key])
                    try:
                        meta[key] = f'models/{absolute.relative_to(models_root).as_posix()}'
                    except ValueError:
                        meta[key] = 'models'
                zf.writestr(arcname, json.dumps(meta, indent=2))
            else:
                zf.write(src, arcname)
    log.info('model bundle written: %s (%d files, %.1f MB)',
             bundle_path, len(members), bundle_path.stat().st_size / 1e6)
    return bundle_path


def resolve_bundle(source: str | Path, workdir: Path) -> Path:
    """Extract (if zipped) and absolutize the bundle's internal paths.
    Returns a directory usable as `model_run_dir` by the inference pipeline."""
    source = Path(source)
    if source.is_dir():
        root = source
    else:
        root = Path(workdir) / 'model_bundle'
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as zf:
            zf.extractall(root)
    for name, keys in _BUNDLE_META_PATH_KEYS.items():
        meta_path = root / name
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        for key in keys:
            if not Path(meta[key]).is_absolute():
                meta[key] = str((root / meta[key]).resolve())
        meta_path.write_text(json.dumps(meta, indent=2))
    if not (root / 's2_meta.json').exists():
        raise ValueError(
            f'модельный бандл {source} не содержит s2_meta.json — это не артефакт '
            f'обучения этой ноды (ожидается zip, созданный режимом train)')
    return root


# ------------------------------------------------------- product contract

def _load_s3_meta(bundle_root: Path):
    p = bundle_root / 's3_meta.json'
    if not p.exists():
        return None
    from ars.data.stages_meta import S2Meta, S3Meta
    raw = json.loads(p.read_text())
    inner = dict(raw['s2_meta'])
    for k in ('epi_latent_mean', 'epi_latent_std', 'sem_latent_mean', 'sem_latent_std'):
        inner[k] = tuple(inner[k]) if inner[k] is not None else None
    raw['s2_meta'] = S2Meta(**inner)
    for k in ('class_names', 'base_kinds', 'metaparams'):
        raw[k] = tuple(raw[k])
    raw['feature_layout'] = dict(raw['feature_layout'])
    return S3Meta(**raw)


def build_product_contract(scored, spans_path: str, recast_flag: bool,
                           s3_meta=None) -> tuple[Any, str]:
    """The legacy end2end contract: (anomaly_traces pandas frame,
    test_anomalies JSON string) — flagged traces only, enriched with trace
    time bounds, user query/response, classifier label when available."""
    import polars as pl
    from ars.main import Anomalies, extract_query_response
    from ars.specification.spec import recast as do_recast

    detected = scored.filter(pl.col('detector_is_anomaly')).drop('detector_is_anomaly')
    if s3_meta is not None and detected.height:
        from ars.stages.s3__classifier import classify_anomalies
        detected = classify_anomalies(detected.lazy(), s3_meta).collect()

    lf_spans = pl.scan_parquet(spans_path)
    if recast_flag:
        lf_spans = do_recast(lf_spans)
    if detected.height:
        query_resp = extract_query_response(lf_spans, detected).collect()
        detected = detected.join(query_resp, on='trace_id', how='left')
    if 'rca_report_str' in detected.columns:
        detected = detected.rename({'rca_report_str': 'rca_results'})
    enriched = Anomalies.enrich(detected, Anomalies.bounds(lf_spans)) if detected.height \
        else detected.with_columns(
            pl.lit('').alias('starttime'), pl.lit('').alias('endtime'),
            pl.lit(0).alias('confidence'))
    payload = json.dumps({'anomalies': Anomalies.records(enriched) if enriched.height else []},
                         ensure_ascii=False)
    return enriched.to_pandas(), payload


# ----------------------------------------------------------------- modes

def _read_html_reports(s1_meta, s2_meta) -> dict:
    read = lambda p: Path(p).read_text(encoding='utf-8') if Path(p).exists() else ''
    return {
        'data':     read(Path(s1_meta.output_dir) / 'data_report.html'),
        'detector': read(Path(s2_meta.output_dir) / 'summary_report.html'),
    }


def run_train(cfg, params: dict[str, Any]) -> dict:
    from laim.pipeline import (Manifest, _apply_runtime, cmd_eval, cmd_infer,
                               cmd_prepare, cmd_train, make_run_dir, setup_logging)
    _apply_runtime(cfg)
    run_dir = make_run_dir(cfg, 'platform_train')
    setup_logging(run_dir, cfg.runtime.log_level)
    manifest = Manifest(run_dir, cfg)
    from laim.runlog import gpu_topology
    manifest.record_metrics('gpu_topology', gpu_topology())

    prep = cmd_prepare(cfg, run_dir, manifest)
    trained = cmd_train(cfg, run_dir, manifest, prep['s1_meta'])
    report = cmd_eval(cfg, run_dir, manifest, prep['s1_meta'], trained['s2_meta'])

    store = Path(params.get('model_store_dir') or '/mnt/data/laim/models')
    store = _writable_store(store)
    bundle = create_bundle(run_dir, store)
    manifest.record_artifact('model_bundle', bundle)

    import pandas as pd
    anomaly_traces = pd.DataFrame()
    test_anomalies = json.dumps({'anomalies': []}, ensure_ascii=False)
    if cfg.paths.infer_spans:
        cmd_infer(cfg, run_dir, manifest, run_dir, cfg.paths.infer_spans)
        import polars as pl
        scored = pl.read_parquet(run_dir / 'detections.parquet')
        anomaly_traces, test_anomalies = build_product_contract(
            scored, cfg.paths.infer_spans, cfg.runtime.recast,
            s3_meta=_load_s3_meta(run_dir))

    manifest_data = json.loads((run_dir / 'manifest.json').read_text())
    s3_metrics = manifest_data['metrics'].get('train_classifier', {})
    return {
        'model_out':                    str(bundle),
        'detector_metrics_holdout':     {**report['test']['overall'],
                                         'threshold': report['threshold'],
                                         'best_experiment': report['best_experiment']},
        'classifier_metrics_holdout':   s3_metrics.get('test_metrics_as_reported', s3_metrics),
        'eval_report':                  json.loads((run_dir / 'eval_report.json').read_text()),
        'anomaly_traces':               anomaly_traces,
        'test_anomalies':               test_anomalies,
        'html_reports':                 _read_html_reports(prep['s1_meta'], trained['s2_meta']),
        'manifest':                     manifest_data,
    }


def run_inference(cfg, params: dict[str, Any]) -> dict:
    from laim.pipeline import (Manifest, _apply_runtime, cmd_infer, make_run_dir,
                               setup_logging)
    _apply_runtime(cfg)
    if not cfg.paths.infer_spans:
        raise ValueError('режим inference: подключите порт path_traces_infer '
                         '(parquet со спанами для скоринга)')
    source = params.get('model_in') or params.get('model_path')
    if not source:
        raise ValueError('режим inference: подключите порт model_in (бандл из режима '
                         'train) или задайте параметр model_path')

    run_dir = make_run_dir(cfg, 'platform_infer')
    setup_logging(run_dir, cfg.runtime.log_level)
    manifest = Manifest(run_dir, cfg)
    from laim.runlog import gpu_topology
    manifest.record_metrics('gpu_topology', gpu_topology())
    manifest.record_input('model_bundle', source)

    bundle_root = resolve_bundle(source, run_dir)

    # embedding params must match training — the bundle's manifest is the truth
    bundle_manifest = json.loads((bundle_root / 'manifest.json').read_text()) \
        if (bundle_root / 'manifest.json').exists() else {}
    trained_data_cfg = bundle_manifest.get('config', {}).get('data', {})
    from laim.config import load_config
    cfg = load_config(None, [
        f'paths.infer_spans={cfg.paths.infer_spans}',
        f'paths.embedder={cfg.paths.embedder}',
        f'paths.output_root={cfg.paths.output_root}',
        f'runtime.device={cfg.runtime.device}',
        f'runtime.seed={cfg.runtime.seed}',
        f'runtime.recast={cfg.runtime.recast}',
        f'eval.attribution_top_k={cfg.eval.attribution_top_k}',
    ] + [f'data.{k}={trained_data_cfg[k]}'
         for k in ('embedding_batch_size', 'embedding_max_length')
         if k in trained_data_cfg])

    cmd_infer(cfg, run_dir, manifest, bundle_root, cfg.paths.infer_spans)

    import polars as pl
    scored = pl.read_parquet(run_dir / 'detections.parquet')
    anomaly_traces, test_anomalies = build_product_contract(
        scored, cfg.paths.infer_spans, cfg.runtime.recast,
        s3_meta=_load_s3_meta(bundle_root))

    manifest_data = json.loads((run_dir / 'manifest.json').read_text())
    return {
        'model_out':                    str(source),
        'detector_metrics_holdout':     {},   # no labels at inference
        'classifier_metrics_holdout':   {},
        'eval_report':                  {'n_traces_scored': scored.height,
                                         'n_detected': int(scored['detector_is_anomaly'].sum()),
                                         'n_truncated': int(scored['detector_truncated'].sum())},
        'anomaly_traces':               anomaly_traces,
        'test_anomalies':               test_anomalies,
        'html_reports':                 {},
        'manifest':                     manifest_data,
    }


def run_node(**params: Any) -> dict:
    """Entry point called by the platform via run.py::main(**params)."""
    _align_thread_env()               # must run before the first polars import
    _quarantine_broken_torchaudio()   # must run before the first transformers import
    # HOME=/ on the platform: the default HF cache (~/.cache/huggingface) is
    # unwritable there; local model loads can still touch it lazily
    if not os.environ.get('HF_HOME'):
        os.environ['HF_HOME'] = str(Path(tempfile.gettempdir()) / 'laim' / 'hf')
    # JAX and the torch embedder pool SHARE the GPUs on this node. JAX's
    # default preallocation (80% of every visible device, legacy
    # c0__env_setup) would starve the encoding workers — grow on demand
    # instead. An operator value already in the environment wins.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

    from laim.runlog import gpu_topology
    topo = gpu_topology()
    if topo.get('available'):
        for g in topo['gpus']:
            log.info('gpu %d: %s | %d MiB total, %d MiB in use | driver %s',
                     g['index'], g['name'], g['memory_total_mib'],
                     g['memory_used_mib'], g['driver'])
        log.info('gpu memory policy: XLA_PYTHON_CLIENT_PREALLOCATE=%s '
                 'MEM_FRACTION=%s CUDA_VISIBLE_DEVICES=%s',
                 topo['xla_preallocate'], topo['xla_mem_fraction'],
                 topo['cuda_visible_devices'] or '<all>')
    else:
        log.info('no GPUs visible to nvidia-smi (%s)', topo.get('reason', 'n/a'))

    params = _normalize_port_params(params)
    mode = str(params.get('mode') or 'train').strip().lower()
    cfg = build_config(params)
    if mode == 'train':
        if not cfg.paths.train_spans:
            raise ValueError('режим train: подключите порт path_traces_train')
        return run_train(cfg, params)
    if mode == 'inference':
        return run_inference(cfg, params)
    raise ValueError(f'неизвестный режим: {mode!r} (допустимо: train | inference)')
