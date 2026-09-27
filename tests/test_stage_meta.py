"""Stage metas (AUDIT_05 F-83): the selection value is the VAL value the choice
was made on; the TEST value is stored separately; legacy bundles still load."""
import dataclasses
from types import SimpleNamespace

import pytest

from ars.data.stages_meta import S2Meta, S3Meta


def _results(name: str, val: float, test: float) -> dict:
    return {'experiment': name,
            'val_metrics': {'youden': val, 'f1': val},
            'test_metrics': {'youden': test, 'f1': test}}


def test_s2_meta_keeps_selection_and_test_values_apart(tmp_path):
    """F-83 FIXED: best_metric_value held the TEST metric although the choice
    was made on VAL (F-02) — and s3 fed it to the classifier as a feature."""
    from ars.configuration.c2__detector import S2Config
    from ars.stages.s2__detector import _build_s2_meta
    from ars.tools.utilities.miscellaneous import FileIO

    exp_dir = tmp_path / 'expB'
    exp_dir.mkdir()
    FileIO.pickle_write(exp_dir / 'combined_model.pkl', {
        'config': SimpleNamespace(sz_latent_epi=4, sz_latent_sem=8),
        'best_threshold': 0.5, 'normalize_latent': False,
        'epi_latent_mean': None, 'epi_latent_std': None,
        'sem_latent_mean': None, 'sem_latent_std': None, 'calibration': {}})
    cfg = S2Config(s1_meta=None, output_dir=tmp_path, output_prefix='t', run_id=None)
    data = SimpleNamespace(max_len=3, epi_dim=2, sem_dim=8)
    # expA wins on TEST, expB on VAL: selection must follow VAL
    meta = _build_s2_meta(cfg, data, (_results('expA', 0.2, 0.9), _results('expB', 0.6, 0.3)))
    assert meta.best_experiment == 'expB'
    assert meta.select_on == 'val'
    assert meta.selection_value == pytest.approx(0.6)
    assert meta.test_value == pytest.approx(0.3)
    assert not hasattr(meta, 'best_metric_value')


def _s2_dict(**extra) -> dict:
    base = {f.name: None for f in dataclasses.fields(S2Meta)}
    base.update(output_dir='o', experiment_dir='e', best_experiment='x', select_metric='youden',
                max_len=1, epi_dim=1, sem_dim=1, epi_sz_latent=1, sem_sz_latent=1,
                seq_pad_chunk=8, best_threshold=0.5, normalize_latent=False,
                epi_latent_mean=[0.0], test_metrics={}, calibration={})
    for k in ('select_on', 'selection_value', 'test_value'):
        base.pop(k)
    base.update(extra)
    return base


def test_legacy_s2_meta_json_still_loads():
    """Bundles written before F-83 carry `best_metric_value` (the TEST value).
    It maps to test_value, and s3 keeps using it as its metaparameter, so an
    old classifier sees the same constant it was trained with."""
    meta = S2Meta.from_dict(_s2_dict(best_metric_value=0.7))
    assert meta.test_value == 0.7 and meta.selection_value is None and meta.select_on == 'val'
    assert meta.selection_value_or_legacy == 0.7
    assert meta.epi_latent_mean == (0.0,)


def test_new_s2_meta_round_trips_and_s3_uses_the_selection_value():
    from ars.stages.s3__classifier import Features
    raw = _s2_dict(select_on='val', selection_value=0.6, test_value=0.3)
    meta = S2Meta.from_dict(dataclasses.asdict(S2Meta.from_dict(raw)))
    assert (meta.selection_value, meta.test_value) == (0.6, 0.3)
    assert Features.metaparams(meta)[1] == pytest.approx(0.6)


def test_legacy_s3_meta_json_still_loads():
    s3 = {f.name: None for f in dataclasses.fields(S3Meta)}
    for k in ('select_on', 'selection_value', 'test_value'):
        s3.pop(k)
    s3.update(output_dir='o', experiment_dir='e', best_experiment='logreg', select_metric='f1',
              best_metric_value=0.4, n_classes=2, class_names=['a', 'b'], feature_dim=3,
              feature_layout={'z_epi': 1}, base_kinds=['lr'], meta_solver='logreg', n_folds=5,
              test_metrics={}, metaparams=[1.0], s2_meta=_s2_dict(best_metric_value=0.7))
    meta = S3Meta.from_dict(s3)
    assert meta.test_value == 0.4 and meta.selection_value is None
    assert meta.s2_meta.test_value == 0.7
    assert meta.class_names == ('a', 'b') and meta.metaparams == (1.0,)
