"""Experiment grid (AUDIT_05 F-77, F-78): metrics come from the config, codes
are parsed rather than filtered against the compiled catalogue."""
import json
from pathlib import Path

import pytest

from ars.configuration.experiments.e2__detector import (
    CODES, EXPERIMENTS, build_grid, parse_code, resolve_grid)
from tests.test_embeddings import standin_embedder  # noqa: F401  (session fixture)


def test_default_grid_carries_no_metric():
    """F-77 FIXED: build_grid used to cycle youden/precision/recall/f1 by list
    position, so detector.threshold_metric was never applied (the deep
    experiment always thresholded by precision)."""
    assert [e().name for e in EXPERIMENTS] == list(CODES)
    for exp in EXPERIMENTS:
        assert exp().threshold_metric is None and exp().select_metric is None
        assert exp().threshold_metric_or('f1') == 'f1'


def test_per_experiment_metric_is_an_explicit_override_only():
    grid = build_grid(('hub_mse_mse_08_4', 'hub_mse_hub_16_4'),
                      metrics={'hub_mse_hub_16_4': 'precision'})
    assert [e().threshold_metric_or('youden') for e in grid] == ['youden', 'precision']


def test_code_outside_catalogue_runs_exactly_that_experiment():
    """F-78 FIXED: codes outside CODES were dropped silently by a filter over
    the compiled grid (PLAN's hub_mse_hub_16_4 would not have run)."""
    assert 'hub_mse_hub_16_4' not in CODES
    grid = resolve_grid(['hub_mse_hub_16_4'])
    assert [e().name for e in grid] == ['hub_mse_hub_16_4']
    exp = grid[0]()
    assert exp.batch_size == 16 and exp.learning_rate == pytest.approx(1e-4)
    assert exp.get_epi_loss_params()[0] == 'huber'
    assert exp.get_combined_loss_params()[0] == 'huber'


def test_requested_codes_keep_order_and_collapse_duplicates():
    grid = resolve_grid(['hub_mse_mse_08_4', 'hub_mse_hub_16_4',
                         'hub_mse_hub_32_4_deep', 'hub_mse_mse_08_4'])
    assert [e().name for e in grid] == ['hub_mse_mse_08_4', 'hub_mse_hub_16_4',
                                        'hub_mse_hub_32_4_deep']


def test_none_keeps_the_compiled_grid():
    assert resolve_grid(None) == EXPERIMENTS


@pytest.mark.parametrize('code', [
    'hub_mse_mse_08',            # too few parts
    'hub_mse_foo_08_4',          # unknown loss
    'hub_mse_mse_x8_4',          # batch not an integer
    'hub_mse_mse_00_4',          # batch not positive
    'hub_mse_mse_08_e4',         # lr exponent not an integer
    'hub_mse_mse_08_4_huge',     # unknown architecture
])
def test_malformed_code_names_the_expected_format(code):
    with pytest.raises(ValueError, match='Формат'):
        parse_code(code)
    with pytest.raises(ValueError, match='Формат'):
        resolve_grid([code])


def test_empty_request_is_an_error():
    with pytest.raises(ValueError, match='пуст'):
        resolve_grid([])


def test_pipeline_rejects_bad_detector_config_before_s1():
    from laim.config import load_config
    from laim.pipeline import check_detector_config
    check_detector_config(load_config())                     # defaults are valid
    with pytest.raises(ValueError, match='Формат'):
        check_detector_config(load_config(None, ['detector.experiments=["hub_mse_mse"]']))
    with pytest.raises(ValueError, match='threshold_metric'):
        check_detector_config(load_config(None, ['detector.threshold_metric=auc']))


def test_bad_config_fails_before_a_run_directory_exists(tmp_path):
    from laim.config import load_config
    from laim.pipeline import run
    root = tmp_path / 'runs'
    for bad in ('detector.experiments=["hub_mse_mse"]',
                'data.injection_fractions={"halucination": 0.1}'):
        with pytest.raises(ValueError):
            run(load_config(None, [bad, f'paths.output_root={root}']), 'train')
    assert not root.exists()


@pytest.mark.slow
def test_threshold_metric_from_config_reaches_every_experiment(
        standin_embedder, fixture_spans, tmp_path):  # noqa: F811
    """F-77 + F-78 end to end: detector.threshold_metric=f1 is the threshold
    metric of EVERY experiment, and a code outside CODES trains too."""
    from laim.config import load_config
    from laim.pipeline import run

    train_path = tmp_path / 'fixture_train.parquet'
    fixture_spans.write_parquet(train_path)
    codes = ['hub_mse_mse_08_4', 'hub_mse_hub_08_5']          # the second is not in CODES
    cfg = load_config(None, [
        f'paths.train_spans={train_path}',
        f'paths.embedder={standin_embedder}',
        f'paths.output_root={tmp_path / "runs"}',
        'detector.epochs=1',
        f'detector.experiments={json.dumps(codes)}',
        'detector.threshold_metric=f1',
        'classifier.enabled=false',
    ])
    run_dir = Path(run(cfg, 'train')['run_dir'])
    s1_meta = json.loads((run_dir / 's1_meta.json').read_text())
    models = Path(s1_meta['output_dir']) / 'models'
    for code in codes:
        results = json.loads((models / code / 'results.json').read_text())
        assert results['threshold_metric'] == 'f1'
        assert results['config']['threshold_metric'] == 'f1'

