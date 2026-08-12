"""Pipeline orchestration over the legacy `ars` stages.

Phase 4 rule: this module WIRES, it does not reimplement numerics. Each stage
function calls the same `ars` entry points as `ars/main.py::main` so behavior is
provably identical (see AUDIT_04_parity.md); it adds config, logging, manifests
and a real evaluation surface around them.

Training and inference are separate functions with separate artifacts:
  prepare+train  → run_dir with models + meta
  infer          → takes a finished run_dir + a spans parquet, writes scores
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path, PurePath
from typing import Any

from laim.config import RunConfig
from laim.runlog import Manifest, StageTimer, setup_logging

log = logging.getLogger('laim.pipeline')


def make_run_dir(cfg: RunConfig, kind: str) -> Path:
    base = Path(cfg.paths.output_root)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = base / f'{stamp}_{kind}_{cfg.config_hash()}'
    n = 1
    while run_dir.exists():
        run_dir = base / f'{stamp}_{kind}_{cfg.config_hash()}_{n}'
        n += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _apply_runtime(cfg: RunConfig) -> None:
    import ars.configuration.c0__env_setup  # noqa: F401  (env side effects, legacy)
    from ars.configuration.c0__device import Device
    from ars.configuration.c0__env_setup import Runtime
    Runtime.apply(track_peak=False, disable_progress=not cfg.runtime.progress,
                  progress_every=0.0)
    Device.of(cfg.runtime.device).force()


def cmd_synth(cfg: RunConfig, run_dir: Path, manifest: Manifest) -> dict:
    from ars.data.synthesis import GenConfig, synthesize
    out_path = Path(cfg.synth.out_path or (run_dir / 'synthetic_spans.parquet'))
    with StageTimer(manifest, 'synth', log):
        frame = synthesize(GenConfig(target_spans=cfg.synth.target_spans,
                                     seed=cfg.synth.seed)).collect()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(out_path)
    stats = {'spans': frame.height,
             'traces': frame['trace_id'].n_unique(),
             'agents': frame['agent_id'].n_unique()}
    manifest.record_metrics('synth', stats)
    manifest.record_artifact('synthetic_spans', out_path)
    log.info('synth: %s -> %s', stats, out_path)
    return {'out_path': str(out_path), **stats}


def cmd_validate(cfg: RunConfig, run_dir: Path, manifest: Manifest,
                 spans_path: str | None = None) -> dict:
    import polars as pl
    from ars.data.validation import Quality
    from ars.specification.spec import recast
    path = spans_path or cfg.paths.train_spans
    manifest.record_input('validate_spans', path)
    with StageTimer(manifest, 'validate', log):
        lf = pl.scan_parquet(path)
        if cfg.runtime.recast:
            lf = recast(lf)
        q = Quality(lf)
        verdict = q.verdict().collect().to_dicts()[0]
        tagged = q.tagged().collect()
        by_trace = tagged.group_by('trace_id').agg(pl.col('rejected').any())
        n_traces = by_trace.height
        n_rejected = int(by_trace['rejected'].sum())
    result = {
        'verdict': {k: (bool(v) if isinstance(v, bool) else int(v)) for k, v in verdict.items()},
        'traces_total': n_traces,
        'traces_rejected': n_rejected,
        'trace_rejection_rate': round(n_rejected / n_traces, 6) if n_traces else None,
    }
    manifest.record_metrics('validate', result)
    log.info('validate: %s', result)
    return result


def cmd_prepare(cfg: RunConfig, run_dir: Path, manifest: Manifest) -> dict:
    """s1: spans parquet → features → injection → traces → split → normalize."""
    from ars.stages.s1__data import process_train_data
    manifest.record_input('train_spans', cfg.paths.train_spans)
    manifest.record_input('embedder', Path(cfg.paths.embedder) / 'config.json')
    run_id = f's{cfg.runtime.seed}'
    with StageTimer(manifest, 'prepare', log):
        s1_cfg, s1_meta = process_train_data(
            PurePath(cfg.paths.train_spans),
            PurePath(cfg.paths.embedder),
            PurePath(run_dir), 'laim', run_id,
            recast=cfg.runtime.recast,
            overrides={**cfg.to_s1_overrides(), 'device': cfg.runtime.device
                       if cfg.runtime.device != 'gpu' else 'cuda'})
    from dataclasses import asdict
    meta_dict = asdict(s1_meta)
    manifest.record_artifact('s1_output_dir', s1_meta.output_dir)
    manifest.record_metrics('prepare', {
        'epi_dim': s1_meta.epi_dim,
        'train_traces': s1_meta.train_samples,
        'val_traces': s1_meta.val_samples,
        'test_traces': s1_meta.test_samples,
        'anomaly_types': list(s1_meta.anomaly_types)})
    (run_dir / 's1_meta.json').write_text(json.dumps(meta_dict, indent=2))
    return {'s1_meta': s1_meta, 's1_cfg': s1_cfg}


def cmd_train(cfg: RunConfig, run_dir: Path, manifest: Manifest, s1_meta) -> dict:
    from ars.stages.s2__detector import train_detector
    from ars.tools.tui.tui import redirect_native_stderr
    with StageTimer(manifest, 'train_detector', log):
        with redirect_native_stderr(PurePath(run_dir) / 'native_debug.log'):
            s2_meta = train_detector(s1_meta, overrides=cfg.to_s2_overrides())
    manifest.record_metrics('train_detector', {
        'best_experiment': s2_meta.best_experiment,
        'best_threshold': s2_meta.best_threshold,
        'test_metrics_as_reported': dict(s2_meta.test_metrics),
        'calibration': dict(s2_meta.calibration)})
    manifest.record_artifact('s2_experiment_dir', s2_meta.experiment_dir)
    from dataclasses import asdict
    (run_dir / 's2_meta.json').write_text(json.dumps(asdict(s2_meta), indent=2))

    s3_meta = None
    if cfg.classifier.enabled:
        from ars.stages.s3__classifier import train_classifier
        try:
            with StageTimer(manifest, 'train_classifier', log):
                with redirect_native_stderr(PurePath(run_dir) / 'native_debug.log'):
                    s3_meta = train_classifier(
                        s1_meta, s2_meta, overrides={'seed': cfg.classifier.seed})
            manifest.record_metrics('train_classifier', {
                'best_experiment': s3_meta.best_experiment,
                'test_metrics_as_reported': dict(s3_meta.test_metrics)})
        except Exception as e:  # F-06: known-crash stage; recorded, not silent
            log.error('classifier stage failed (known finding F-06): %s', e)
            manifest.record_metrics('train_classifier', {'failed': str(e)})
    return {'s2_meta': s2_meta, 's3_meta': s3_meta}


def cmd_eval(cfg: RunConfig, run_dir: Path, manifest: Manifest,
             s1_meta, s2_meta) -> dict:
    """Full evaluation on the s1 val/test splits with the trained detector."""
    import numpy as np
    import polars as pl

    from ars.models.m2__detector.confidence import Predict
    from ars.stages.s2__detector import Pad, load_models_for_inference
    from laim import evaluation

    data_dir = Path(s1_meta.output_dir)
    prefix = f'{s1_meta.prefix}_{s1_meta.run_id}' if s1_meta.run_id else s1_meta.prefix
    splits = {s: pl.read_parquet(data_dir / f'{prefix}_{s}.parquet')
              for s in ('train', 'val', 'test')}
    max_len = s2_meta.max_len
    meta, models = load_models_for_inference(s2_meta)

    def tensors(df: pl.DataFrame):
        epi, epi_m = Pad.split(df, 'epi_sequence', s2_meta.epi_dim, max_len,
                               s2_meta.seq_pad_chunk)
        sem, sem_m = Pad.split(df, 'sem_sequence_sem_vector', s2_meta.sem_dim,
                               max_len, s2_meta.seq_pad_chunk)
        return epi, epi_m, sem, sem_m

    report: dict[str, Any] = {'best_experiment': s2_meta.best_experiment,
                              'threshold': s2_meta.best_threshold}
    with StageTimer(manifest, 'eval', log):
        for name in ('val', 'test'):
            df = splits[name]
            epi, epi_m, sem, sem_m = tensors(df)
            out = Predict.batch(meta, models, epi, epi_m, sem, sem_m)
            y = df['is_anomaly'].fill_null(0).to_numpy().astype(np.int32)
            report[name] = evaluation.evaluate_split(
                y=y,
                score=np.asarray(out.e_comb),
                p_anomaly=np.asarray(out.p_anomaly),
                threshold=s2_meta.best_threshold,
                anomaly_types=df['anomaly_type'].to_numpy()
                if 'anomaly_type' in df.columns else None,
                target_fpr=cfg.eval.target_fpr,
                assumed_prevalence=cfg.eval.assumed_prevalence,
                calibration_bins=cfg.eval.calibration_bins)

        # per-trace latency on test tensors (batch=1)
        epi, epi_m, sem, sem_m = tensors(splits['test'])
        n = epi.shape[0]
        if n:
            def predict_one(i: int) -> None:
                out = Predict.batch(meta, models, epi[i:i+1], epi_m[i:i+1],
                                    sem[i:i+1], sem_m[i:i+1])
                float(out.p_anomaly[0])  # force device sync
            report['latency_per_trace'] = {
                **evaluation.measure_latency(predict_one, n, cfg.eval.latency_reps),
                'device': cfg.runtime.device, 'batch': 1}

    (run_dir / 'eval_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str))
    manifest.record_metrics('eval', {
        'test_overall': report['test']['overall'],
        'test_recall_at_fpr': report['test'][f'recall_at_fpr_{cfg.eval.target_fpr}'],
        'test_calibration_ece': report['test']['calibration']['ece'],
        'latency': report.get('latency_per_trace')})
    manifest.record_artifact('eval_report', run_dir / 'eval_report.json')
    log.info('eval: test overall %s', report['test']['overall'])
    return report


def cmd_infer(cfg: RunConfig, run_dir: Path, manifest: Manifest,
              model_run_dir: Path, spans_path: str) -> dict:
    """Score a spans parquet with a previously trained run (separate from train)."""
    import polars as pl

    from ars.data.stages_meta import S1Meta, S2Meta
    from ars.stages.s1__data import prepare_test_data, process_train_data  # noqa: F401
    from ars.stages.s2__detector import detect_anomalies

    s1_raw = json.loads((model_run_dir / 's1_meta.json').read_text())
    s1_raw['epi_features'] = tuple(s1_raw['epi_features'])
    s1_raw['anomaly_types'] = tuple(s1_raw['anomaly_types'])
    s1_raw['raw_files'] = tuple(s1_raw['raw_files'])
    if isinstance(s1_raw['epi_normalization'], str):
        # legacy default=str serialization fallback
        import ast
        s1_raw['epi_normalization'] = ast.literal_eval(s1_raw['epi_normalization'])
    s1_meta = S1Meta(**s1_raw)

    raw = json.loads((model_run_dir / 's2_meta.json').read_text())
    for k in ('epi_latent_mean', 'epi_latent_std', 'sem_latent_mean', 'sem_latent_std'):
        raw[k] = tuple(raw[k]) if raw[k] is not None else None
    s2_meta = S2Meta(**raw)

    from ars.configuration.c1__data import S1Config
    from ars.tools.tui.tui_data import ColorSchemeDataScienceSakura
    s1_cfg = S1Config(
        input_parquet_files=(PurePath(spans_path),),
        output_dir=PurePath(run_dir) / 'infer_features',
        output_prefix='laim',
        embedder_path=PurePath(cfg.paths.embedder),
        output_color_scheme=ColorSchemeDataScienceSakura(),
        recast=cfg.runtime.recast,
        **cfg.to_s1_overrides())

    manifest.record_input('infer_spans', spans_path)
    with StageTimer(manifest, 'infer', log):
        lf = prepare_test_data(s1_cfg, PurePath(spans_path), s1_meta,
                               PurePath(run_dir), 'laim')
        # F-27: full audit trail — every trace scored, detections flagged
        scored = detect_anomalies(lf, s2_meta, only_anomalies=False).collect()
    out_path = run_dir / 'detections.parquet'
    scored.write_parquet(out_path)
    manifest.record_artifact('detections', out_path)
    result = {'n_traces_scored': scored.height,
              'n_detected': int(scored['detector_is_anomaly'].sum()),
              'n_truncated': int(scored['detector_truncated'].sum()),
              'out_path': str(out_path)}
    manifest.record_metrics('infer', result)
    log.info('infer: %s', result)
    return result


def run(cfg: RunConfig, command: str, spans: str | None = None,
        model_dir: str | None = None) -> dict:
    run_dir = make_run_dir(cfg, command)
    setup_logging(run_dir, cfg.runtime.log_level)
    manifest = Manifest(run_dir, cfg)
    log.info('run dir: %s | command: %s | config hash: %s',
             run_dir, command, cfg.config_hash())
    _apply_runtime(cfg)

    if command == 'synth':
        return {'run_dir': str(run_dir), **cmd_synth(cfg, run_dir, manifest)}
    if command == 'validate':
        return {'run_dir': str(run_dir), **cmd_validate(cfg, run_dir, manifest, spans)}
    if command == 'infer':
        if not (model_dir and (spans or cfg.paths.infer_spans)):
            raise SystemExit('infer requires --model-dir and --spans (or paths.infer_spans)')
        return {'run_dir': str(run_dir),
                **cmd_infer(cfg, run_dir, manifest, Path(model_dir),
                            spans or cfg.paths.infer_spans)}
    if command in ('prepare', 'train', 'eval', 'all'):
        prep = cmd_prepare(cfg, run_dir, manifest)
        if command == 'prepare':
            return {'run_dir': str(run_dir)}
        trained = cmd_train(cfg, run_dir, manifest, prep['s1_meta'])
        if command in ('eval', 'all'):
            cmd_eval(cfg, run_dir, manifest, prep['s1_meta'], trained['s2_meta'])
        if command == 'all' and (spans or cfg.paths.infer_spans):
            cmd_infer(cfg, run_dir, manifest, run_dir,
                      spans or cfg.paths.infer_spans)
        return {'run_dir': str(run_dir)}
    raise SystemExit(f'unknown command: {command}')
