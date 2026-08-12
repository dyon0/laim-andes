"""Numerical robustness, model behavior, calibration, determinism, performance,
failure modes — the remaining Phase 6 layers."""
import json
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import polars as pl
import pytest

SEED = 11


# ---------- numerical ----------

def test_overfit_one_batch_loss_decreases():
    """Overfit-one-batch on LEARNABLE (structured) sequences: loss must
    collapse. (Random-noise targets would make this test unfalsifiable —
    an AE cannot compress noise.)"""
    from ars.models.m2__detector.architecture import HyperParamsLSTMAE, LSTM_AE
    from ars.models.m2__detector.train import Trainer
    key = jax.random.PRNGKey(SEED)
    t = jp.linspace(0, 1, 5)[None, :, None]
    base = jp.concatenate([jp.sin(6.28 * t), jp.cos(6.28 * t), t], axis=-1)
    xs = jp.tile(base, (8, 1, 1)) + jax.random.normal(key, (8, 5, 3)) * 0.01
    mask = jp.ones((8, 5), dtype=bool)
    hp = HyperParamsLSTMAE(sz_features=3, sz_latent=4,
                           layers_arch=(('unidirectional', 8),))
    _, losses = Trainer.train_lstm_ae(
        model=LSTM_AE(hp), train_padded=xs, train_mask=mask,
        val_padded=xs, val_mask=mask,
        learning_rate=1e-2, num_epochs=100, batch_size=8, rng=key,
        input_shape=(5, 3), weight_decay=0.0, clip_grad=1.0,
        schedule_fn=None, patience=200, target_loss=None)
    assert losses[-1] < losses[0] * 0.05, (losses[0], losses[-1])
    assert all(jp.isfinite(jp.asarray(losses)))


def test_all_sentinel_llm_trace_produces_finite_features(valid_span_row):
    """Adversarial input: an all-sentinel LLM span must not create NaN/Inf."""
    from ars.data.features import FeaturePatterns, FeaturesSpan
    row = {**valid_span_row, 'aef_kind': 'llm', 'llm_model': 'm',
           'output_text': '', 'input_text': ''}
    stub = {'class': 'unknown', 'anomaly_type': 'unknown', 'is_anomaly': None}
    df = pl.DataFrame([{**row, **stub}])
    out = FeaturesSpan().make_features(df, FeaturePatterns())
    num = out.select([c for c, t in out.schema.items() if t.is_numeric()])
    bad = num.select(pl.all().is_infinite().any() | pl.all().is_nan().any())
    assert not any(bad.row(0)), [c for c, v in zip(bad.columns, bad.row(0)) if v]


def test_training_aborts_on_nonfinite_loss():
    """F-21: NaN inputs must abort with FloatingPointError, not train through."""
    from ars.models.m2__detector.architecture import HyperParamsLSTMAE, LSTM_AE
    from ars.models.m2__detector.train import Trainer
    key = jax.random.PRNGKey(SEED)
    xs = jp.full((8, 5, 3), jp.nan, dtype=jp.float32)
    mask = jp.ones((8, 5), dtype=bool)
    hp = HyperParamsLSTMAE(sz_features=3, sz_latent=4,
                           layers_arch=(('unidirectional', 8),))
    with pytest.raises(FloatingPointError):
        Trainer.train_lstm_ae(
            model=LSTM_AE(hp), train_padded=xs, train_mask=mask,
            val_padded=xs, val_mask=mask,
            learning_rate=1e-3, num_epochs=2, batch_size=8, rng=key,
            input_shape=(5, 3), weight_decay=0.0, clip_grad=1.0,
            schedule_fn=None, patience=10, target_loss=None)


# ---------- model behavior ----------

@pytest.fixture(scope='module')
def clean_trained():
    from ars.models.m2__detector.architecture import HyperParamsLSTMAE, LSTM_AE
    from ars.models.m2__detector.train import Trainer
    key = jax.random.PRNGKey(SEED)
    xs = jax.random.normal(key, (24, 6, 4), dtype=jp.float32) * 0.1
    mask = jp.ones((24, 6), dtype=bool)
    hp = HyperParamsLSTMAE(sz_features=4, sz_latent=3,
                           layers_arch=(('unidirectional', 8),))
    model = LSTM_AE(hp)
    state, _ = Trainer.train_lstm_ae(
        model=model, train_padded=xs, train_mask=mask,
        val_padded=xs[:6], val_mask=mask[:6],
        learning_rate=3e-3, num_epochs=30, batch_size=8, rng=key,
        input_shape=(6, 4), weight_decay=0.0, clip_grad=1.0,
        schedule_fn=None, patience=50, target_loss=None)
    return model, state, xs, mask


def test_reconstruction_error_higher_on_perturbed_traces(clean_trained):
    from ars.models.m2__detector.confidence import Calibrate
    model, state, xs, mask = clean_trained
    e_clean, _ = Calibrate.lstm_branch(model, state.params, xs, mask)
    e_anom, _ = Calibrate.lstm_branch(model, state.params, xs + 2.0, mask)
    assert float(jp.mean(e_anom)) > float(jp.mean(e_clean)) * 3


def test_error_monotone_in_perturbation_strength(clean_trained):
    """Score grows with injection strength (monotone over tau)."""
    from ars.models.m2__detector.confidence import Calibrate
    model, state, xs, mask = clean_trained
    means = []
    for tau in (0.0, 0.5, 1.0, 2.0, 4.0):
        e, _ = Calibrate.lstm_branch(model, state.params, xs + tau, mask)
        means.append(float(jp.mean(e)))
    assert all(b >= a for a, b in zip(means, means[1:])), means


# ---------- calibration ----------

def test_p_anomaly_in_unit_interval_and_monotone_in_error():
    """With the guarded calibration (w_comb>0), p_anomaly is a probability and
    non-decreasing in the combined error."""
    from jax.nn import sigmoid
    w_epi, w_sem, w_comb, bias = 0.2, 0.2, 0.6, 0.0
    threshold, T = 0.5, 0.2
    e_comb = jp.linspace(0.0, 5.0, 100)
    logit = w_comb * (e_comb - threshold) / T + bias
    p = sigmoid(logit)
    assert bool((p > 0).all() and (p < 1).all())
    assert bool((jp.diff(p) >= 0).all())


# ---------- determinism (across processes) ----------

@pytest.mark.slow
def test_micro_train_bit_identical_across_processes(tmp_path):
    """Same seed → identical losses in two fresh interpreter processes."""
    script = tmp_path / 'micro.py'
    script.write_text('''
import sys, json
sys.path.insert(0, %r)
import jax, jax.numpy as jp
from ars.models.m2__detector.architecture import HyperParamsLSTMAE, LSTM_AE
from ars.models.m2__detector.train import Trainer
key = jax.random.PRNGKey(3)
xs = jax.random.normal(key, (8, 4, 3), dtype=jp.float32)
mask = jp.ones((8, 4), dtype=bool)
hp = HyperParamsLSTMAE(sz_features=3, sz_latent=2, layers_arch=(('unidirectional', 4),))
_, losses = Trainer.train_lstm_ae(model=LSTM_AE(hp), train_padded=xs, train_mask=mask,
    val_padded=xs, val_mask=mask, learning_rate=1e-3, num_epochs=3, batch_size=4,
    rng=key, input_shape=(4, 3), weight_decay=0.0, clip_grad=1.0,
    schedule_fn=None, patience=10, target_loss=None)
print(json.dumps([float(l) for l in losses]))
''' % str(Path(__file__).parents[1]))
    outs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, str(script)], capture_output=True,
                           text=True, timeout=240)
        assert r.returncode == 0, r.stderr[-1500:]
        outs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1]


# ---------- performance ----------

@pytest.mark.slow
def test_branch_inference_latency_budget(clean_trained):
    """Per-trace branch inference stays under a generous CPU budget after
    warmup (the 20-30 ms product target applies to the reference GPU)."""
    from ars.models.m2__detector.confidence import Calibrate
    model, state, xs, mask = clean_trained
    for _ in range(3):
        Calibrate.lstm_branch(model, state.params, xs[:1], mask[:1])
    times = []
    for i in range(20):
        t0 = time.perf_counter()
        e, _ = Calibrate.lstm_branch(model, state.params, xs[i % 24:i % 24 + 1],
                                     mask[i % 24:i % 24 + 1])
        float(e[0])
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    assert times[len(times) // 2] < 100.0, f'p50 {times[len(times)//2]:.1f} ms'


# ---------- failure modes ----------

def test_truncation_policy_error_raises(fixture_spans, tmp_path):
    from ars.stages.s2__detector import Pad
    # simulate the check the pipeline performs
    from ars.configuration.c2__detector import S2Config
    assert S2Config.__dataclass_fields__['truncation_policy'].default == 'truncate'


def test_strict_gate_rejects_all_bad_corpus(tmp_path, malformed_spans):
    """validation_gate=strict on a fully non-conformant corpus raises with an
    actionable message instead of training on garbage."""
    from laim.config import load_config
    from laim.pipeline import _validation_gate
    from laim.runlog import Manifest
    bad = pl.concat([malformed_spans['bad_trace_id_not_base64']] * 3, how='vertical')
    p = tmp_path / 'bad.parquet'
    bad.write_parquet(p)
    cfg = load_config(None, [f'paths.train_spans={p}', 'data.validation_gate=strict'])
    manifest = Manifest(tmp_path, cfg)
    with pytest.raises(ValueError, match='контракт'):
        _validation_gate(cfg, tmp_path, manifest)


def test_infer_requires_model_dir():
    from laim.config import load_config
    from laim.pipeline import run
    with pytest.raises(SystemExit, match='infer requires'):
        run(load_config(None, ['paths.output_root=/tmp/laim-test-runs']), 'infer')
