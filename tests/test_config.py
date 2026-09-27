"""Config layer: defaults ← TOML ← CLI overrides, round-trip and coercion."""
import pytest

from laim.config import RunConfig, load_config


def test_defaults_load():
    cfg = load_config()
    assert cfg.detector.select_on == 'val'
    assert cfg.data.validation_gate == 'warn'


def test_toml_and_cli_precedence(tmp_path):
    f = tmp_path / 'c.toml'
    f.write_text('[runtime]\nseed = 7\n\n[detector]\nepochs = 3\n')
    cfg = load_config(f, ['detector.epochs=99'])
    assert cfg.runtime.seed == 7          # from TOML
    assert cfg.detector.epochs == 99      # CLI wins


def test_list_override_parses_json():
    cfg = load_config(None, ['detector.experiments=["hub_mse_mse_08_4"]'])
    assert cfg.detector.experiments == ('hub_mse_mse_08_4',)


def test_scalar_string_for_list_field_is_rejected():
    with pytest.raises(ValueError, match='expects a list'):
        load_config(None, ['detector.experiments=oops'])


def test_bool_and_float_coercion():
    cfg = load_config(None, ['runtime.recast=false', 'eval.target_fpr=0.2'])
    assert cfg.runtime.recast is False
    assert cfg.eval.target_fpr == 0.2


def test_config_hash_stable_and_sensitive():
    a, b = RunConfig(), RunConfig()
    assert a.config_hash() == b.config_hash()
    c = load_config(None, ['runtime.seed=1'])
    assert c.config_hash() != a.config_hash()


def test_unknown_key_is_rejected_with_a_suggestion():
    """F-82 FIXED: unknown keys were skipped silently (no effect, no error)."""
    with pytest.raises(ValueError, match='detector.typo'):
        load_config(None, ['detector.typo=1'])
    with pytest.raises(ValueError, match='did you mean: detector.epochs'):
        load_config(None, ['detector.epoch=5'])
    with pytest.raises(ValueError, match='did you mean: detector'):
        load_config(None, ['detectr.epochs=5'])


def test_unknown_key_in_toml_is_rejected(tmp_path):
    f = tmp_path / 'c.toml'
    f.write_text('[data]\nmax_corelation = 0.9\n')
    with pytest.raises(ValueError, match='data.max_correlation'):
        load_config(f)


def test_section_given_a_scalar_is_rejected():
    with pytest.raises(ValueError, match='config section'):
        load_config(None, ['detector=5'])


def test_shipped_configs_load():
    from pathlib import Path
    for f in sorted((Path(__file__).parents[1] / 'configs').glob('*.toml')):
        load_config(f)
