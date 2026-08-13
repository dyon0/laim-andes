"""SberDS platform node: descriptor sanity, param mapping, bundle round-trip,
and (slow) the full train→bundle→inference hand-off through run.py::main."""
import json
import zipfile
from pathlib import Path

import polars as pl
import pytest

from laim import platform
from tests.test_embeddings import standin_embedder  # noqa: F401  (session fixture)

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------- descriptor

@pytest.fixture(scope='module')
def descriptor() -> dict:
    return json.loads((REPO / 'descriptor.json').read_text())


def test_descriptor_entry_point(descriptor):
    rc = descriptor['script']['runConfiguration']
    assert rc['sourceFiles'] == ['run.py']
    assert rc['functionName'] == 'main'
    import run as run_module
    assert callable(run_module.main)


def test_descriptor_out_ports_match_adapter(descriptor):
    out_ports = [p['name'] for p in descriptor['ports'] if not p['in']]
    assert tuple(out_ports) == platform.OUT_PORTS


def test_descriptor_in_ports_are_known_and_correctly_required(descriptor):
    in_ports = {p['name']: p for p in descriptor['ports'] if p['in']}
    assert set(in_ports) == {'path_traces_train', 'path_traces_infer',
                             'path_embedder', 'model_in'}
    assert in_ports['path_embedder']['required'] is True
    for name in ('path_traces_train', 'path_traces_infer', 'model_in'):
        assert in_ports[name]['required'] is False, name   # mode-dependent
    for name, p in in_ports.items():
        assert p.get('getPortAsLocalPath') is True, name


def test_descriptor_ui_parameters_are_all_understood(descriptor):
    known = set(platform.PARAM_MAP) | set(platform.NODE_PARAMS)
    for page in descriptor['ui']['settings']:
        for comp in page['components']:
            for field in comp.get('config', {}).get('components', []):
                if 'parameter' in field:
                    assert field['parameter'] in known, field['parameter']


def test_descriptor_gpu_option_present(descriptor):
    assert descriptor['script']['baseImageKey'] == 'py312-gpu'
    device_fields = [f for page in descriptor['ui']['settings']
                     for comp in page['components']
                     for f in comp.get('config', {}).get('components', [])
                     if f.get('parameter') == 'device']
    values = [list(v)[0] for v in device_fields[0]['allowedValues']]
    assert set(values) == {'cpu', 'gpu'}


# ----------------------------------------------------------- param mapping

def test_build_config_maps_params_and_ports(tmp_path):
    cfg = platform.build_config({
        'mode': 'train',
        'device': 'gpu',
        'seed': 7,
        'epochs': 3,
        'experiments': '["hub_mse_mse_08_4"]',
        'validation_gate': 'strict',
        'path_traces_train': '/tmp/a.parquet',
        'path_traces_infer': '/tmp/b.parquet',
        'path_embedder': '/models/emb',
        'config_overrides': 'detector.cal_min_pos=9; data.max_correlation=0.99',
    })
    assert cfg.runtime.device == 'gpu' and cfg.runtime.seed == 7
    assert cfg.detector.epochs == 3
    assert cfg.detector.experiments == ('hub_mse_mse_08_4',)
    assert cfg.data.validation_gate == 'strict'
    assert cfg.paths.train_spans == '/tmp/a.parquet'
    assert cfg.paths.infer_spans == '/tmp/b.parquet'
    assert cfg.paths.embedder == '/models/emb'
    assert cfg.data.max_correlation == 0.99


def test_build_config_ignores_empty_params():
    cfg = platform.build_config({'device': '', 'seed': None})
    assert cfg.runtime.device == 'cpu' and cfg.runtime.seed == 12345


def test_resolve_embedder_unzips(tmp_path):
    model = tmp_path / 'm'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    z = tmp_path / 'emb.zip'
    with zipfile.ZipFile(z, 'w') as zf:
        zf.write(model / 'config.json', 'USER-bge-m3/config.json')
    resolved = Path(platform._resolve_embedder(str(z)))
    assert (resolved / 'config.json').exists()
    assert platform._resolve_embedder(str(model)) == str(model)


# --------------------------------------------------------- bundle handling

def _fake_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / 'run'
    models = run_dir / 'traces_train' / 'models'
    exp = models / 'expA'
    exp.mkdir(parents=True)
    (exp / 'combined_model.pkl').write_bytes(b'\x80\x04N.')
    (models / 'best').mkdir()
    (models / 'best' / 'best_info.json').write_text('{"best_experiment": "expA"}')
    (run_dir / 's1_meta.json').write_text('{"fake": 1}')
    (run_dir / 's2_meta.json').write_text(json.dumps({
        'output_dir': str(models), 'experiment_dir': str(exp)}))
    (run_dir / 'manifest.json').write_text('{"config": {"data": {}}}')
    return run_dir


def test_bundle_round_trip(tmp_path):
    run_dir = _fake_run_dir(tmp_path)
    bundle = platform.create_bundle(run_dir, tmp_path / 'store')
    assert bundle.exists() and bundle.suffix == '.zip'

    root = platform.resolve_bundle(bundle, tmp_path / 'work')
    meta = json.loads((root / 's2_meta.json').read_text())
    assert Path(meta['experiment_dir']).is_absolute()
    assert (Path(meta['experiment_dir']) / 'combined_model.pkl').exists()
    assert (root / 's1_meta.json').exists()


def test_resolve_bundle_rejects_non_bundle(tmp_path):
    junk = tmp_path / 'junk.zip'
    with zipfile.ZipFile(junk, 'w') as zf:
        zf.writestr('readme.txt', 'not a model')
    with pytest.raises(ValueError, match='s2_meta'):
        platform.resolve_bundle(junk, tmp_path / 'w')


# ------------------------------------------------------ end-to-end (slow)

@pytest.mark.slow
def test_platform_train_then_inference(standin_embedder, fixture_spans, tmp_path):  # noqa: F811
    import run as run_module

    train_path = tmp_path / 'train.parquet'
    fixture_spans.write_parquet(train_path)
    infer_ids = fixture_spans.select('trace_id').unique().sort('trace_id').head(4)
    infer_path = tmp_path / 'infer.parquet'
    fixture_spans.join(infer_ids, on='trace_id', how='semi').write_parquet(infer_path)

    result = run_module.main(
        mode='train',
        path_traces_train=str(train_path),
        path_embedder=str(standin_embedder),
        model_store_dir=str(tmp_path / 'store'),
        output_root=str(tmp_path / 'runs'),
        epochs=1, experiments='["hub_mse_mse_08_4"]',
        classifier_enabled=False, latency_reps=3, seed=12345,
    )
    assert set(result) == set(platform.OUT_PORTS)
    assert Path(result['model_out']).exists()
    assert result['detector_metrics_holdout']['best_experiment'] == 'hub_mse_mse_08_4'
    assert 'per_anomaly_type' in result['eval_report']['test']
    assert result['html_reports']['data']          # s1 HTML present
    assert result['manifest']['config_hash']

    inference = run_module.main(
        mode='inference',
        model_in=result['model_out'],
        path_traces_infer=str(infer_path),
        path_embedder=str(standin_embedder),
        output_root=str(tmp_path / 'runs'),
        seed=12345,
    )
    assert set(inference) == set(platform.OUT_PORTS)
    assert inference['eval_report']['n_traces_scored'] == 4
    payload = json.loads(inference['test_anomalies'])
    assert 'anomalies' in payload
    for rec in payload['anomalies']:      # legacy product contract fields
        assert {'trace_id', 'starttime', 'endtime', 'confidence',
                'anomaly_type'} <= set(rec)
    df = inference['anomaly_traces']
    assert hasattr(df, 'to_dict')          # pandas frame for the dataframe port


def test_inference_requires_model():
    import run as run_module
    with pytest.raises(ValueError, match='model_in'):
        run_module.main(mode='inference', path_traces_infer='/tmp/x.parquet',
                        path_embedder='/tmp/emb')
