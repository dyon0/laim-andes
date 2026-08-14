"""SberDS platform node: descriptor sanity, param mapping, bundle round-trip,
and (slow) the full train→bundle→inference hand-off through run.py::main."""
import json
import sys
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
    # Data ports: type "dataframe" and NO getPortAsLocalPath — the platform
    # stores dataframe ports as parquet parts with POSITIONAL column names
    # and applies the real schema only when parsing the port itself (run
    # 2026-08-13 22:16: raw-file delivery had columns "0".."57").
    for name in ('path_traces_train', 'path_traces_infer'):
        assert in_ports[name]['type'] == 'dataframe', name
        assert 'getPortAsLocalPath' not in in_ports[name], name
    # Model ports: blobs, delivered as local files
    for name in ('path_embedder', 'model_in'):
        assert in_ports[name]['type'] == 'default', name
        assert in_ports[name].get('getPortAsLocalPath') is True, name


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


def test_resolve_embedder_extensionless_blob(tmp_path):
    """SberDS delivers model ports as an extension-less file (observed name
    `unstructured_data`, ZIP by magic bytes) — resolution must sniff content."""
    model = tmp_path / 'm'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    blob = tmp_path / 'unstructured_data'
    with zipfile.ZipFile(blob, 'w') as zf:
        zf.write(model / 'config.json', 'USER-bge-m3/config.json')
    resolved = Path(platform._resolve_embedder(str(blob)))
    assert (resolved / 'config.json').exists()


def test_resolve_embedder_tar_blob(tmp_path):
    import tarfile
    model = tmp_path / 'm'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    blob = tmp_path / 'unstructured_data'
    with tarfile.open(blob, 'w:gz') as tf:
        tf.add(model / 'config.json', 'USER-bge-m3/config.json')
    resolved = Path(platform._resolve_embedder(str(blob)))
    assert (resolved / 'config.json').exists()


def test_resolve_embedder_real_st_layout_skips_module_dirs(tmp_path):
    """Regression for run 2026-08-13 23:20: a real SentenceTransformer archive
    contains numbered module dirs (1_Pooling/config.json); resolution must
    return the MODEL root (modules.json marker), never the pooling module —
    lexically '1_Pooling/config.json' sorts before 'config.json'."""
    blob = tmp_path / 'unstructured_data'
    with zipfile.ZipFile(blob, 'w') as zf:
        zf.writestr('USER-bge-m3/config.json', '{"model_type": "xlm-roberta"}')
        zf.writestr('USER-bge-m3/modules.json', '[]')
        zf.writestr('USER-bge-m3/1_Pooling/config.json', '{"word_embedding_dimension": 1024}')
    resolved = Path(platform._resolve_embedder(str(blob)))
    assert resolved.name == 'USER-bge-m3'
    assert (resolved / 'modules.json').exists()

    # without modules.json the numbered module dir must still be skipped
    blob2 = tmp_path / 'blob2'
    with zipfile.ZipFile(blob2, 'w') as zf:
        zf.writestr('m/1_Pooling/config.json', '{}')
        zf.writestr('m/config.json', '{"model_type": "xlm-roberta"}')
    resolved2 = Path(platform._resolve_embedder(str(blob2)))
    assert resolved2.name == 'm'


def test_resolve_embedder_rejects_unknown_blob(tmp_path):
    blob = tmp_path / 'unstructured_data'
    blob.write_bytes(b'\x00\x01\x02\x03 definitely not an archive')
    with pytest.raises(ValueError, match='path_embedder'):
        platform._resolve_embedder(str(blob))


# ----------------------------------------------- directory-shaped data ports

def _parted_spans(tmp_path: Path, n_parts: int = 3) -> Path:
    """Emulate a SberDS dataframe port: a directory of part-*.snappy.parquet."""
    df = pl.read_parquet(REPO / 'data' / 'traces_1k_sample.parquet')
    port = tmp_path / 'port_dir'
    port.mkdir()
    step = df.height // n_parts + 1
    for i in range(n_parts):
        df.slice(i * step, step).write_parquet(
            port / f'part-{i:05d}-c000.snappy.parquet')
    (port / '_SUCCESS').write_text('')   # spark marker must be ignored
    return port


def test_spans_scan_source_directory_port(tmp_path):
    from laim.config import spans_scan_source
    port = _parted_spans(tmp_path)
    src = spans_scan_source(port)
    df = pl.scan_parquet(src).collect()
    ref = pl.read_parquet(REPO / 'data' / 'traces_1k_sample.parquet')
    assert df.height == ref.height
    # plain files and globs pass through untouched
    f = REPO / 'data' / 'traces_1k_sample.parquet'
    assert spans_scan_source(f) == str(f)
    assert spans_scan_source(src) == src


def test_file_fingerprint_directory_and_glob(tmp_path):
    from laim.runlog import file_fingerprint
    port = _parted_spans(tmp_path)
    fp = file_fingerprint(port)
    assert fp['exists'] and fp['mode'] == 'name-size-manifest'
    assert fp['files'] == 4   # 3 parts + _SUCCESS
    assert fp['bytes'] > 0
    fp2 = file_fingerprint(port)
    assert fp2['sha256'] == fp['sha256']   # deterministic
    from laim.config import spans_scan_source
    fp_glob = file_fingerprint(spans_scan_source(port))
    assert fp_glob['exists'] and fp_glob['files'] == 3   # glob excludes _SUCCESS


def _platform_mangled_frame(n_rows: int = 40):
    """Reproduce what the platform hands over when a dataframe port is parsed
    in-memory: a pandas DataFrame whose Boolean columns were cast to strings
    ('!!! WARNING !!! Column llm_stream is casted from bool to string' in the
    platform log), plus an unknown _trash_ column that must pass through."""
    df = pl.read_parquet(REPO / 'data' / 'traces_1k_sample.parquet').head(n_rows)
    df = df.with_columns(
        pl.Series('llm_stream', [i % 2 == 0 for i in range(n_rows)]))  # mixed values
    pdf = df.to_pandas()
    for col in ('llm_profanity_check', 'llm_stream',
                'session_id_derived', 'session_id_generated'):
        pdf[col] = pdf[col].astype(str)                     # True -> 'True'
    pdf['_trash_llm_stream'] = pdf['llm_stream']
    return df, pdf


def test_dataframe_port_staged_and_dtypes_repaired(tmp_path):
    """An in-memory pandas payload is written back to parquet with every
    contract column restored to its contract dtype via the spec Recast."""
    original, pdf = _platform_mangled_frame()
    staged = platform._stage_dataframe_port('path_traces_train', pdf)

    out = pl.read_parquet(staged)
    assert out.height == original.height
    for col in ('llm_profanity_check', 'llm_stream',
                'session_id_derived', 'session_id_generated'):
        assert out[col].dtype == pl.Boolean, col
    assert out['llm_stream'].to_list() == original['llm_stream'].to_list()
    assert out['trace_id'].to_list() == original['trace_id'].to_list()
    assert out['_trash_llm_stream'].dtype == pl.Utf8   # unknown column untouched


def test_normalize_port_params_paths_pass_through():
    params = {'path_traces_train': '/some/dir', 'path_embedder': '/emb',
              'model_in': None, 'mode': 'train'}
    assert platform._normalize_port_params(params) == params


def test_normalize_port_params_rejects_dataframe_on_model_ports():
    import pandas as pd
    with pytest.raises(ValueError, match='model_in'):
        platform._normalize_port_params({'model_in': pd.DataFrame({'a': [1]})})


def test_build_config_accepts_dataframe_payload(tmp_path):
    """End to end through the exact crash site of run 2026-08-13 21:52:
    a pandas DataFrame in path_traces_train must yield a scannable source."""
    _, pdf = _platform_mangled_frame()
    params = platform._normalize_port_params(
        {'path_traces_train': pdf, 'mode': 'train'})
    cfg = platform.build_config(params)
    scanned = pl.scan_parquet(cfg.paths.train_spans).collect()
    assert scanned.height == 40
    assert scanned['llm_stream'].dtype == pl.Boolean


def test_core_schema_preflight_rejects_positional_columns(tmp_path):
    """Raw SberDS dataframe-port parts carry positional column names — the
    preflight must fail with the actionable hint BEFORE recast can
    sentinel-fill 46 columns and 'train' on one fake trace (run 22:16)."""
    from laim.pipeline import ensure_core_spans_columns
    alien = pl.DataFrame({str(i): [1.0, 2.0] for i in range(58)}).with_columns(
        pl.lit('x').alias('class'), pl.lit('y').alias('anomaly_type'))
    f = tmp_path / 'alien.parquet'
    alien.write_parquet(f)
    with pytest.raises(ValueError, match='getPortAsLocalPath'):
        ensure_core_spans_columns(str(f))


def test_core_schema_preflight_accepts_real_spans_and_names_missing(tmp_path):
    from laim.pipeline import ensure_core_spans_columns
    ensure_core_spans_columns(str(REPO / 'data' / 'traces_1k_sample.parquet'))

    partial = pl.DataFrame({'trace_id': ['t1'], 'other': [1]})
    f = tmp_path / 'partial.parquet'
    partial.write_parquet(f)
    with pytest.raises(ValueError, match='agent_id'):
        ensure_core_spans_columns(str(f))


def test_backend_mismatch_rule():
    """The image pairs torchaudio+xpu with torch+cu128 — only that inconsistent
    pairing gets quarantined; consistent CUDA/XPU/CPU pairings do not."""
    assert platform._backend_mismatch('2.8.0+xpu', '2.8.0+cu128') is True
    assert platform._backend_mismatch('2.8.0+cu128', '2.8.0+cu128') is False
    assert platform._backend_mismatch('2.8.0+xpu', '2.8.0+xpu') is False
    assert platform._backend_mismatch('2.8.0', '2.8.0+cu128') is False
    assert platform._backend_mismatch('2.8.0', '2.13.0') is False


def test_quarantine_blocks_transformers_availability_check(monkeypatch):
    """sys.modules[name] = None is the documented block marker: find_spec —
    what transformers' is_torchaudio_available() consults — must return None,
    and a direct import must raise cleanly instead of dlopen-crashing."""
    import importlib.metadata
    import importlib.util

    real_version = importlib.metadata.version
    fake = {'torchaudio': '2.8.0+xpu', 'torch': '2.8.0+cu128'}
    monkeypatch.setattr(importlib.metadata, 'version',
                        lambda name: fake.get(name) or real_version(name))
    monkeypatch.delitem(sys.modules, 'torchaudio', raising=False)
    try:
        platform._quarantine_broken_torchaudio()
        assert sys.modules['torchaudio'] is None
        assert importlib.util.find_spec('torchaudio') is None
        with pytest.raises(ImportError):
            import torchaudio  # noqa: F401
    finally:
        sys.modules.pop('torchaudio', None)


def test_quarantine_leaves_consistent_pairing_alone(monkeypatch):
    import importlib.metadata
    real_version = importlib.metadata.version
    fake = {'torchaudio': '2.8.0+cu128', 'torch': '2.8.0+cu128'}
    monkeypatch.setattr(importlib.metadata, 'version',
                        lambda name: fake.get(name) or real_version(name))
    monkeypatch.delitem(sys.modules, 'torchaudio', raising=False)
    platform._quarantine_broken_torchaudio()
    assert 'torchaudio' not in sys.modules


def test_writable_store_falls_back(tmp_path, monkeypatch):
    import tempfile as _tempfile
    monkeypatch.setattr(_tempfile, 'gettempdir', lambda: str(tmp_path / 'tmp'))
    # a file where a directory is expected raises OSError even for root
    # (chmod-based denial would not: tests may run as root)
    blocker = tmp_path / 'denied'
    blocker.write_text('')
    got = platform._writable_store(blocker / 'models')
    assert got == tmp_path / 'tmp' / 'laim' / 'models'
    ok = platform._writable_store(tmp_path / 'ok')
    assert ok == tmp_path / 'ok' and ok.is_dir()


# --------------------------------------------------------- bundle handling

def _fake_run_dir(tmp_path: Path, with_s3: bool = False) -> Path:
    """The real artifact layout: s2 models/ and s3 classifier/ are SIBLINGS
    under the s1 data dir (that sibling-ness is what run 2026-08-14 12:21
    tripped over — single-root packing lost the s3 stack)."""
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
    if with_s3:
        cls = run_dir / 'traces_train' / 'classifier'
        cls_exp = cls / 'logreg_stack'
        cls_exp.mkdir(parents=True)
        (cls_exp / 'stack.pkl').write_bytes(b'\x80\x04N.')
        (run_dir / 's3_meta.json').write_text(json.dumps({
            'output_dir': str(cls), 'experiment_dir': str(cls_exp)}))
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


def test_bundle_packs_the_s3_classifier_stack(tmp_path):
    """Run 2026-08-14 12:21: inference scored 875 traces and died on a
    missing models/stack.pkl — s3 artifacts live in classifier/, a SIBLING
    of the s2 models/ root, and were never packed. Both roots must land in
    the archive under their own prefixes, and the resolved s3 meta must
    point at a real stack.pkl."""
    run_dir = _fake_run_dir(tmp_path, with_s3=True)
    bundle = platform.create_bundle(run_dir, tmp_path / 'store')

    with zipfile.ZipFile(bundle) as zf:
        names = set(zf.namelist())
    assert 'models/expA/combined_model.pkl' in names
    assert 'models_s3/logreg_stack/stack.pkl' in names

    root = platform.resolve_bundle(bundle, tmp_path / 'work')
    s3 = json.loads((root / 's3_meta.json').read_text())
    assert Path(s3['experiment_dir']).is_absolute()
    assert (Path(s3['experiment_dir']) / 'stack.pkl').exists()
    s2 = json.loads((root / 's2_meta.json').read_text())
    assert (Path(s2['experiment_dir']) / 'combined_model.pkl').exists()


def test_bundle_refuses_to_ship_without_the_stack(tmp_path):
    """A trained classifier whose stack.pkl is missing must fail at TRAIN
    time — never after a long scoring run."""
    run_dir = _fake_run_dir(tmp_path, with_s3=True)
    (run_dir / 'traces_train' / 'classifier' / 'logreg_stack' / 'stack.pkl').unlink()
    with pytest.raises(ValueError, match='stack.pkl'):
        platform.create_bundle(run_dir, tmp_path / 'store')


def test_bundle_refuses_meta_path_outside_its_root(tmp_path):
    """The old code silently rewrote unmappable paths to 'models' — the
    exact mechanism that hid the missing stack. Now it is a hard error."""
    run_dir = _fake_run_dir(tmp_path, with_s3=True)
    s3 = json.loads((run_dir / 's3_meta.json').read_text())
    s3['experiment_dir'] = str(tmp_path / 'elsewhere')
    (run_dir / 's3_meta.json').write_text(json.dumps(s3))
    with pytest.raises(ValueError, match='вне корня'):
        platform.create_bundle(run_dir, tmp_path / 'store')


def test_classification_failure_does_not_void_scoring(tmp_path, caplog):
    """The production failure path itself: an s3 meta whose experiment_dir
    does not exist (old bundle). The detected frame must come back unlabeled
    with the error in notes — not raise."""
    import logging
    from types import SimpleNamespace
    detected = pl.DataFrame({'trace_id': ['t1', 't2']})
    notes: dict = {}
    dead_meta = SimpleNamespace(experiment_dir=str(tmp_path / 'gone'))
    with caplog.at_level(logging.ERROR, logger='laim.platform'):
        out = platform._classified(detected, dead_meta, notes)
    assert out.equals(detected)                      # scoring survives
    assert 'FileNotFoundError' in notes['classifier_error']
    assert 'WITHOUT classifier labels' in caplog.text

    # no s3 in the bundle: untouched frame, no notes
    notes2: dict = {}
    assert platform._classified(detected, None, notes2).equals(detected)
    assert notes2 == {}


# ------------------------------------------- bundle transport via port bytes

def test_bundle_payload_carries_the_zip_bytes(tmp_path):
    """model_out must carry the bundle BYTES: run 2026-08-14 08:51 proved a
    model_out->model_in wire delivers only the JSON payload (75 bytes — the
    path string, dead with the train container)."""
    bundle = platform.create_bundle(_fake_run_dir(tmp_path), tmp_path / 'store')
    payload = platform.bundle_payload(bundle)
    assert payload['bundle_format'] == 'laim-bundle-b64/1'
    assert payload['filename'] == bundle.name
    assert payload['size_bytes'] == bundle.stat().st_size
    assert payload['stored_path'] == str(bundle)
    import base64
    assert base64.b64decode(payload['bundle_b64']) == bundle.read_bytes()


def test_resolve_bundle_from_port_payload_file(tmp_path):
    """The exact platform delivery: pywrapper json.dump()s the payload to an
    extension-less file; model_in hands us that file's path."""
    bundle = platform.create_bundle(_fake_run_dir(tmp_path), tmp_path / 'store')
    payload_file = tmp_path / 'tmpabc123.model_out'   # platform naming
    payload_file.write_text(json.dumps(platform.bundle_payload(bundle)))
    bundle.unlink()   # the train container is gone — bytes must suffice

    root = platform.resolve_bundle(payload_file, tmp_path / 'work')
    assert (root / 's2_meta.json').exists()
    meta = json.loads((root / 's2_meta.json').read_text())
    assert Path(meta['experiment_dir']).is_absolute()


def test_resolve_bundle_detects_corrupted_transfer(tmp_path):
    bundle = platform.create_bundle(_fake_run_dir(tmp_path), tmp_path / 'store')
    payload = platform.bundle_payload(bundle)
    import base64
    payload['bundle_b64'] = base64.b64encode(b'garbage' * 100).decode()
    f = tmp_path / 'p.model_out'
    f.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='контрольная сумма'):
        platform.resolve_bundle(f, tmp_path / 'work')


def test_resolve_bundle_legacy_dead_path_payload_is_actionable(tmp_path):
    """A payload from the OLD node version: a JSON string with a path into
    the dead train container. Must fail with instructions, not BadZipFile."""
    f = tmp_path / 'tmpxyz.model_out'
    f.write_text(json.dumps('/tmp/laim/models/laim_model_gone.zip'))
    with pytest.raises(ValueError, match='model_path|train'):
        platform.resolve_bundle(f, tmp_path / 'work')

    # dict payload without bytes and with a dead stored_path: same story
    f2 = tmp_path / 'tmp2.model_out'
    f2.write_text(json.dumps({'bundle_b64': '', 'stored_path': '/gone/x.zip'}))
    with pytest.raises(ValueError, match='model_path|train'):
        platform.resolve_bundle(f2, tmp_path / 'work')


def test_resolve_bundle_payload_with_live_stored_path(tmp_path):
    """Shared-storage setups: payload without bytes but with a reachable
    stored_path must still resolve (the >256 MB escape hatch)."""
    bundle = platform.create_bundle(_fake_run_dir(tmp_path), tmp_path / 'store')
    f = tmp_path / 'p.model_out'
    f.write_text(json.dumps({'bundle_b64': '', 'stored_path': str(bundle)}))
    root = platform.resolve_bundle(f, tmp_path / 'work')
    assert (root / 's2_meta.json').exists()


def test_resolve_bundle_rejects_garbage_file(tmp_path):
    f = tmp_path / 'noise.model_out'
    f.write_bytes(b'\x00\x01\x02\x03 not json not zip')
    with pytest.raises(ValueError, match='не распознан'):
        platform.resolve_bundle(f, tmp_path / 'work')


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
    assert result['model_out']['bundle_format'] == 'laim-bundle-b64/1'
    assert Path(result['model_out']['stored_path']).exists()
    assert result['model_out']['bundle_b64']
    assert result['detector_metrics_holdout']['best_experiment'] == 'hub_mse_mse_08_4'
    assert 'per_anomaly_type' in result['eval_report']['test']
    assert result['html_reports']['data']          # s1 HTML present
    assert result['manifest']['config_hash']

    # the hand-off exactly as the platform does it: pywrapper json.dump()s the
    # model_out payload to a file, and model_in delivers that file's path.
    # The original zip is DELETED first — in production the train container
    # is gone by the time inference runs; the port bytes must suffice.
    handoff = tmp_path / 'tmpplatform.model_out'
    handoff.write_text(json.dumps(result['model_out']))
    Path(result['model_out']['stored_path']).unlink()

    inference = run_module.main(
        mode='inference',
        model_in=str(handoff),
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
