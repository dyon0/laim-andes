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
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePath
from typing import Any

log = logging.getLogger('laim.platform')

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


def _resolve_embedder(raw: str | None) -> str | None:
    """Port may deliver a directory or a zip; zips are extracted once."""
    if not raw:
        return None
    p = Path(raw)
    if p.suffix.lower() != '.zip':
        return str(p)
    target = Path(tempfile.mkdtemp(prefix='embedder_'))
    with zipfile.ZipFile(p) as zf:
        zf.extractall(target)
    entries = [e for e in target.iterdir() if e.is_dir()]
    # a zip either contains the model at its root or as a single subdirectory
    return str(entries[0] if len(entries) == 1 and not (target / 'config.json').exists()
               else target)


def build_config(params: dict[str, Any]):
    from laim.config import load_config
    overrides = [f'{PARAM_MAP[k]}={params[k]}'
                 for k in PARAM_MAP if params.get(k) not in (None, '')]
    extra = str(params.get('config_overrides') or '')
    overrides += [line.strip() for line in extra.replace(';', '\n').splitlines()
                  if line.strip() and not line.strip().startswith('#')]
    cfg = load_config(None, overrides)

    port_overrides = []
    if params.get('path_traces_train'):
        port_overrides.append(f"paths.train_spans={params['path_traces_train']}")
    if params.get('path_traces_infer'):
        port_overrides.append(f"paths.infer_spans={params['path_traces_infer']}")
    embedder = _resolve_embedder(params.get('path_embedder'))
    if embedder:
        port_overrides.append(f'paths.embedder={embedder}')
    return load_config(None, overrides + port_overrides)


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

    prep = cmd_prepare(cfg, run_dir, manifest)
    trained = cmd_train(cfg, run_dir, manifest, prep['s1_meta'])
    report = cmd_eval(cfg, run_dir, manifest, prep['s1_meta'], trained['s2_meta'])

    store = Path(params.get('model_store_dir') or '/mnt/data/laim/models')
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
    mode = str(params.get('mode') or 'train').strip().lower()
    cfg = build_config(params)
    if mode == 'train':
        if not cfg.paths.train_spans:
            raise ValueError('режим train: подключите порт path_traces_train')
        return run_train(cfg, params)
    if mode == 'inference':
        return run_inference(cfg, params)
    raise ValueError(f'неизвестный режим: {mode!r} (допустимо: train | inference)')
