"""Integration: raw AEF spans → trained detector → p_anomaly, on CPU.

Uses the 20-trace fixture and 1-epoch training — this validates wiring and
contracts, not model quality.
"""
import json
from pathlib import Path

import polars as pl
import pytest

from tests.test_embeddings import standin_embedder  # noqa: F401  (session fixture)


@pytest.mark.slow
def test_full_pipeline_on_fixture(standin_embedder, fixture_spans, tmp_path):  # noqa: F811
    from laim.config import load_config
    from laim.pipeline import run

    train_path = tmp_path / 'fixture_train.parquet'
    fixture_spans.write_parquet(train_path)
    infer_ids = fixture_spans.select('trace_id').unique().sort('trace_id').head(3)
    infer_path = tmp_path / 'fixture_infer.parquet'
    fixture_spans.join(infer_ids, on='trace_id', how='semi').write_parquet(infer_path)

    cfg = load_config(None, [
        f'paths.train_spans={train_path}',
        f'paths.infer_spans={infer_path}',
        f'paths.embedder={standin_embedder}',
        f'paths.output_root={tmp_path / "runs"}',
        'runtime.seed=12345',
        'detector.epochs=1',
        'detector.experiments=["hub_mse_mse_08_4"]',
        'classifier.enabled=false',
        'eval.latency_reps=5',
    ])
    result = run(cfg, 'all')
    run_dir = Path(result['run_dir'])

    manifest = json.loads((run_dir / 'manifest.json').read_text())
    assert manifest['stages']['prepare']['status'] == 'ok'
    assert manifest['stages']['train_detector']['status'] == 'ok'
    assert manifest['stages']['eval']['status'] == 'ok'
    assert manifest['stages']['infer']['status'] == 'ok'
    assert manifest['config_hash']
    assert manifest['inputs']['train_spans']['sha256']

    report = json.loads((run_dir / 'eval_report.json').read_text())
    for split in ('val', 'test'):
        assert 0 < report[split]['n']
        assert 'per_anomaly_type' in report[split]
        assert 0 <= report[split]['calibration']['ece'] <= 1
    assert report['latency_per_trace']['p50_ms'] > 0

    detections = pl.read_parquet(run_dir / 'detections.parquet')
    assert detections.height == 3                        # audit trail: ALL traces scored
    assert 'detector_p_anomaly' in detections.columns
    assert 'detector_is_anomaly' in detections.columns
    assert 'rca_top_span_indices' in detections.columns  # RCA seam present
    p = detections['detector_p_anomaly']
    assert bool(((p > 0) & (p < 1)).all())
