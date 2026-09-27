"""Inference embeds on the configured device (AUDIT_05 F-76).

cmd_infer built its S1Config without `device`, so every scoring path —
`run.py infer`, `run.py all` with infer_spans, the platform inference mode and
the in-run scoring of path_traces_infer — embedded on CPU even with
runtime.device=gpu, while training embedded on every visible GPU. No GPU is
needed here: the stage config is intercepted before any model loads.
"""
import json
from dataclasses import fields

import pytest

from ars.data.stages_meta import S1Meta, S2Meta
from laim.config import load_config


class _Captured(Exception):
    pass


def _dummy_meta(cls) -> dict:
    tuples = {'raw_files', 'epi_features', 'anomaly_types'}
    dicts = {'epi_normalization', 'semantic_vectors', 'split_config',
             'test_metrics', 'calibration'}
    out = {}
    for f in fields(cls):
        if f.name in tuples:
            out[f.name] = ['x']
        elif f.name in dicts:
            out[f.name] = {}
        elif 'None' in str(f.type):
            out[f.name] = None
        elif f.type in (int, 'int'):
            out[f.name] = 1
        elif f.type in (float, 'float'):
            out[f.name] = 1.0
        elif f.type in (bool, 'bool'):
            out[f.name] = False
        else:
            out[f.name] = 'x'
    return out


@pytest.fixture
def model_run_dir(tmp_path):
    d = tmp_path / 'model'
    d.mkdir()
    (d / 's1_meta.json').write_text(json.dumps(_dummy_meta(S1Meta)))
    (d / 's2_meta.json').write_text(json.dumps(_dummy_meta(S2Meta)))
    return d


@pytest.mark.parametrize('device, expected', [('gpu', 'cuda'), ('cpu', 'cpu')])
def test_cmd_infer_passes_the_device_to_s1(device, expected, model_run_dir,
                                          fixture_spans, tmp_path, monkeypatch):
    import ars.stages.s1__data as s1
    from laim.pipeline import cmd_infer
    from laim.runlog import Manifest

    seen = {}

    def fake_prepare_test_data(cfg, *args, **kwargs):
        seen['cfg'] = cfg
        raise _Captured

    monkeypatch.setattr(s1, 'prepare_test_data', fake_prepare_test_data)
    spans = tmp_path / 'spans.parquet'
    fixture_spans.write_parquet(spans)
    cfg = load_config(None, [f'runtime.device={device}', 'data.embedding_gpus=2'])
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    with pytest.raises(_Captured):
        cmd_infer(cfg, run_dir, Manifest(run_dir, cfg), model_run_dir, str(spans))
    assert seen['cfg'].device == expected
    assert seen['cfg'].embedding_gpus == 2


def test_prepare_and_infer_share_one_device_mapping():
    cfg = load_config(None, ['runtime.device=gpu'])
    assert cfg.s1_device() == 'cuda'
    assert cfg.to_s1_overrides()['device'] == 'cuda'
    assert load_config(None, ['runtime.device=cpu']).to_s1_overrides()['device'] == 'cpu'


def test_platform_inference_honors_the_instance_gpu_settings(tmp_path, monkeypatch):
    """run_inference rebuilds its config from an allow-list; the instance's own
    embedding_gpus / embedding_pool_chunk must survive it (the training run's
    GPU count says nothing about the inference instance)."""
    import laim.pipeline as pipeline
    from laim import platform

    seen = {}

    def fake_cmd_infer(cfg, *args, **kwargs):
        seen['cfg'] = cfg
        raise _Captured

    monkeypatch.setattr(pipeline, '_apply_runtime', lambda cfg: None)
    monkeypatch.setattr(pipeline, 'cmd_infer', fake_cmd_infer)
    monkeypatch.setattr(platform, 'resolve_bundle', lambda source, workdir: tmp_path)
    cfg = load_config(None, [
        'runtime.device=gpu', 'data.embedding_gpus=3', 'data.embedding_pool_chunk=777',
        'paths.infer_spans=/nonexistent.parquet', f'paths.output_root={tmp_path / "runs"}'])
    with pytest.raises(_Captured):
        platform.run_inference(cfg, {'model_path': str(tmp_path)})
    assert seen['cfg'].runtime.device == 'gpu'
    assert seen['cfg'].s1_device() == 'cuda'
    assert seen['cfg'].data.embedding_gpus == 3
    assert seen['cfg'].data.embedding_pool_chunk == 777
